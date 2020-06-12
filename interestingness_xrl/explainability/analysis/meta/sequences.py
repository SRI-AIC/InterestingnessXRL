__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import heapq
import numpy as np
from os.path import join
from interestingness_xrl.explainability.analysis import AnalysisBase
from interestingness_xrl.explainability.analysis.interaction.state_frequency import StateFrequencyAnalysis
from interestingness_xrl.explainability.analysis.task.transitions import TransitionAnalysis
from interestingness_xrl.explainability.analysis.meta.transition_values import TransitionValuesAnalysis
from interestingness_xrl.explainability.analysis.interaction.recency import RecencyAnalysis
from interestingness_xrl.scenarios.scenario_helper import ScenarioHelper
from interestingness_xrl.learning.agents import QValueBasedAgent
from interestingness_xrl.learning import get_features_from_index
from interestingness_xrl.util import print_line


class SequenceAnalysis(AnalysisBase):
    """
    Represents an analysis of common and important sequences of actions according to the agent's history of interaction
    with its environment. In particular, interesting sequences involve starting from a local minima / maxima, frequent,
    uncertain or earlier state, then executing the most likely action, and from then performing actions to reach a local
    maximum (target) state. Only target states that are reachable with a minimum of probability are considered. The most
    valuable target state is chosen as the one with the highest product between the probability and state Q value.
    """

    def __init__(self, helper, agent, freq_analysis, state_action_freq_analysis, transition_analysis,
                 transition_value_analysis, recency_analysis):
        """
        Creates a new sequence analysis.
        :param ScenarioHelper helper: the environment helper containing all necessary config and methods.
        :param QValueBasedAgent agent: the agent to be analyzed.
        :param StateFrequencyAnalysis freq_analysis: the state frequency analysis.
        :param StateActionFrequencyAnalysis state_action_freq_analysis: the state-action frequency analysis.
        :param TransitionAnalysis transition_analysis: the state-action transition analysis.
        :param TransitionValuesAnalysis transition_value_analysis: the value function analysis.
        :param RecencyAnalysis recency_analysis: the state recency analysis.
        """
        super().__init__(helper, agent)

        self.uncertain_future_states = []
        """ The states in which a sequence to any sub-goal is very unlikely (s, n). """

        self.certain_seqs_to_subgoal = []
        """ The likely sequences starting from an interesting state until reaching a certain sub-goal state (s, [(a, s) ,...], prob). """

        self.state_freq_analysis = freq_analysis
        self.state_action_freq_analysis = state_action_freq_analysis
        self.transition_analysis = transition_analysis
        self.transition_value_analysis = transition_value_analysis
        self.recency_analysis = recency_analysis

    def difference_to(self, other):
        """
        Gets the difference between this analysis and another given analysis, i.e., the elements that are new and
        therefore missing in the other analysis. This can be used to highlight what is new in this analysis compared to
        one that was performed at a prior stage of the agent's behavior / learning.
        :param SequenceAnalysis other: the other analysis to get the difference to.
        :return SequenceAnalysis: an analysis resulting from the difference between this and the given analysis.
        """

        # initializes analysis diff
        diff_analysis = SequenceAnalysis(
            self.helper, self.agent, self.state_freq_analysis, self.state_action_freq_analysis,
            self.transition_analysis, self.transition_value_analysis, self.recency_analysis)

        other_uncertain_future_states = set(s for s, *_ in other.uncertain_future_states)
        diff_analysis.uncertain_future_states = \
            [(s, *_) for s, *_ in self.uncertain_future_states
             if s not in other_uncertain_future_states]

        other_certain_seqs_to_subgoal = set(s for s, *_ in other.certain_seqs_to_subgoal)
        diff_analysis.certain_seqs_to_subgoal = \
            [(s, *_) for s, *_ in self.certain_seqs_to_subgoal
             if s not in other_certain_seqs_to_subgoal]

        return diff_analysis

    def analyze(self):

        self.uncertain_future_states = []
        self.certain_seqs_to_subgoal = []

        # first check likely sequences from local minima, uncertain, frequent and earlier states to local maxima states
        source_states = set([x[0] for x in self.transition_value_analysis.local_minima_states])
        # source_states.update([x[0] for x in self.transition_value_analysis.local_maxima_states])
        # source_states.update([x[0] for x in self.transition_value_analysis.val_diff_mean_action_outliers])
        # source_states.update([x[0] for x in self.transition_value_analysis.val_diff_variance_state_outliers])
        # source_states.update([x[0] for x in self.state_freq_analysis.freq_states])
        # source_states.update([x[0] for x in self.state_freq_analysis.infreq_states])
        # source_states.update([x[0] for x in self.state_action_freq_analysis.certain_states])
        # source_states.update([x[0] for x in self.state_action_freq_analysis.uncertain_states])
        # source_states.update([x[0] for x in self.transition_analysis.uncertain_trans])
        # source_states.update([x[0] for x in self.transition_analysis.certain_trans])
        # source_states.update([x[0] for x in self.recency_analysis.earlier_states])
        source_states.add(self.helper.get_terminal_state())

        target_states = set([x[0] for x in self.transition_value_analysis.local_maxima_states])

        next_states = {}
        self._get_best_paths(source_states, target_states, next_states)

        # sorts lists
        self.certain_seqs_to_subgoal.sort(key=lambda e: -len(e[2]))
        self.uncertain_future_states.sort()

    def _get_best_paths(self, source_states, target_states, next_states):

        # for each test/start state
        uncertain_states = set()
        for s in source_states:

            # first executes most likely action and gets next state
            a = int(np.argmax(self.agent.c_sa[s]))
            ns = int(np.argmax(self.agent.c_sas[s][a]))

            if ns in target_states:
                continue

            # tries to get sequence from that state to each target state
            for ts in target_states:
                seq_t = self._get_best_path(ns, ts, next_states)
                a_seq, s_seq, prob = seq_t

                # checks if state next to first needs to be added
                if s != ns:
                    a_seq.insert(0, a)
                    s_seq.insert(0, ns)

                c_s = self.agent.c_s[s].item()
                seq = (s, c_s, list(zip(a_seq, s_seq)), prob)

                if len(a_seq) == 0 or prob == 0:
                    uncertain_states.add(s)
                elif s_seq[-1] in target_states:
                    self.certain_seqs_to_subgoal.append(seq)

        # these states don't have known/observed paths to any of target states
        self.uncertain_future_states.extend([(s, self.agent.c_s[s].item()) for s in uncertain_states])

    def _get_best_path(self, s, ts, next_states):
        """
        Tries to calculate the shortest path (sequence of state-action pairs) from the source state to the given
        target state. Shortest here is most likely, as defined by the agent's action execution probabilities and
        observed state transition probabilities. If no path is found above a given probability, an empty path is
        returned. This method uses a variation of Dijktra's shortest path algorithm as explained in:
        https://algocoding.wordpress.com/2015/03/28/dijkstras-algorithm-part-4a-python-implementation/
        :param int s: the source state.
        :param int ts: the target state.
        :param dict next_states: the dictionary containing the next state information for each state.
        :rtype: tuple
        :return: (a_seq, s_seq, prob[target]) a tuple containing the action and state sequence to reach the target state
        and the probability of reaching it.
        """

        if s == ts:
            return [], [], 0

        predecessors = {}
        prob = {}
        reached_target = False
        terminal_state = self.helper.get_terminal_state()

        prob[s] = 1.
        p_queue = []
        heapq.heappush(p_queue, (-prob[s], s))

        # while there are open states in the queue
        while p_queue:

            # retrieves the next state with highest probability from the queue
            ps_prob, ps = heapq.heappop(p_queue)
            ps_prob = -ps_prob

            if ps_prob == prob[ps]:

                # checks if the target was found, in which case marks as target found
                if ps == ts:
                    reached_target = True
                    break

                # checks if a terminal state is found, continue searching elsewhere
                elif ps == terminal_state:
                    continue

                # otherwise checks all next states from this state
                s_next_states = self._get_next_states(ps, next_states)
                for ns in s_next_states:

                    # calculates probability of reaching next state
                    a, ns_a_prob = s_next_states[ns]
                    ns_prob = ps_prob * ns_a_prob

                    # checks if it's a new next state or we can reach it with a higher probability
                    if ns_prob > 0 and (ns not in prob or ns_prob > prob[ns]):
                        prob[ns] = ns_prob
                        heapq.heappush(p_queue, (-prob[ns], ns))
                        predecessors[ns] = ps, a

        # checks for path to the target state
        if not reached_target:
            return [], [], 0

        # builds the path/action sequence from the source state to the target state
        a_seq = []
        s_seq = []
        ns = ts
        while True:
            ps, a = predecessors[ns]
            s_seq.append(ns)
            a_seq.append(a)
            if ps == s:
                break  # stop if we found the source state
            ns = ps
        a_seq.reverse()
        s_seq.reverse()

        return a_seq, s_seq, prob[ts]

    def _get_next_states(self, s, next_states):

        # checks if already calculated next states
        if s not in next_states:
            next_states[s] = {}

            # checks for unvisited state (observed but no action was ever executed [prob. absorbing state])
            c_s = float(self.agent.c_s[s])
            if c_s == 0:
                return next_states[s]

            # gets visited next states using each action
            for a in range(self.agent.num_actions):
                a_prob = self.agent.c_sa[s][a] / c_s

                next_s = np.nonzero(self.agent.c_sas[s][a])[0].tolist()
                for ns in next_s:
                    if ns == s:
                        continue

                    # gets transition probability
                    ns_prob = self.agent.c_sas[s][a][ns] / float(self.agent.c_sa[s][a])
                    prob = (a_prob * ns_prob).item()

                    # checks transition already added, choose max prob.
                    if ns not in next_states[s] or next_states[s][ns][1] < prob:
                        # adds next state to list with action and transition probability
                        next_states[s][ns] = (a, prob)

        return next_states[s]

    def _save_report(self, file, write_console):

        feats_nbins = self.helper.get_features_bins()
        action_names = self.config.get_action_names()

        print_line('====================================', file, write_console)
        print_line('{} certain sequences to sub-goals found (min. prob > 0):'.format(
            len(self.certain_seqs_to_subgoal)), file, write_console)

        for s, n, seq, prob in self.certain_seqs_to_subgoal:
            self._print_sequence(s, n, seq, prob, feats_nbins, action_names, file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} states with an uncertain future found:'.format(
            len(self.uncertain_future_states)), file, write_console)

        for s, n in self.uncertain_future_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec, True)
            print_line('\t{} (count: {})'.format(feats_labels, n), file, write_console)

    def _print_sequence(self, s, n, seq, prob, feats_nbins, action_names, file, write_console):
        obs_vec = get_features_from_index(s, feats_nbins)
        feats_labels = self.helper.get_features_labels(obs_vec, True)
        print_line('\t{}-{} (count: {}, prob. reaching target: {:.3e})'.format(
            s, feats_labels, n, prob), file, write_console)
        for a, ns in seq:
            obs_vec1 = get_features_from_index(ns, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec1, True)
            obs_vec_diff = self._vec_diff(obs_vec1, obs_vec)
            feats_labels = [feats_labels[i] for i in obs_vec_diff]
            obs_vec = obs_vec1
            print_line('\t\t{} -> {}-{}'.format(action_names[a], ns, feats_labels), file, write_console)
        print_line('____________________________________', file, write_console)

    @staticmethod
    def _vec_diff(obs_vec1, obs_vec2):
        return np.where(obs_vec1 != obs_vec2)[0]

    def get_stats(self):
        return {
            'Num certain seqs to sub-goals': (len(self.certain_seqs_to_subgoal), 0., 1),
            'Num uncertain-future states': (len(self.uncertain_future_states), 0., 1),
            'Max sequence length': (max(len(seq) for _, _, seq, _ in self.certain_seqs_to_subgoal), 0., 1),
        }

    def save_json(self, json_file_path):
        freq_analysis = self.state_freq_analysis
        self.state_freq_analysis = None

        action_freq_analysis = self.state_action_freq_analysis
        self.state_action_freq_analysis = None

        value_analysis = self.transition_value_analysis
        self.transition_value_analysis = None

        recency_analysis = self.recency_analysis
        self.recency_analysis = None

        transition_analysis = self.transition_analysis
        self.transition_analysis = None

        super().save_json(json_file_path)

        self.state_freq_analysis = freq_analysis
        self.state_action_freq_analysis = action_freq_analysis
        self.transition_value_analysis = value_analysis
        self.recency_analysis = recency_analysis
        self.transition_analysis = transition_analysis

    def _save_visual_report(self, path):

        feats_nbins = self.helper.get_features_bins()

        for i, s_seq in enumerate(self.certain_seqs_to_subgoal):
            s, _, seq, _ = s_seq
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'certain-seq-subgoal-#{}-0-s-{}.png'.format(i, s)))
            for j, a_ns in enumerate(seq):
                a, ns = a_ns
                obs_vec = get_features_from_index(ns, feats_nbins)
                self.helper.save_features_image(
                    obs_vec, join(path, 'certain-seq-subgoal-#{}-{}-a-{}-s-{}.png'.format(i, j + 1, a, ns)))

        for s, *_ in self.uncertain_future_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'uncertain-future-s-{}.png'.format(s)))

    def get_sample_interesting_aspects(self, s, a, r, ns):

        aspects = []
        for st, *_ in self.certain_seqs_to_subgoal:
            if st == s:
                aspects.append('certain-seq-subgoal-s-{}'.format(s))
                break

        for st, *_ in self.uncertain_future_states:
            if st == s:
                aspects.append('uncertain-future-s-{}'.format(s))
                break

        return aspects

    def get_interestingness_names(self):
        return ['certain-seq-subgoal-s', 'uncertain-future-s']
