__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
from os.path import join
from interestingness_xrl.explainability import get_pairwise_jensen_shannon_divergence
from interestingness_xrl.explainability.analysis import AnalysisBase
from interestingness_xrl.explainability.analysis.meta.transition_values import TransitionValuesAnalysis
from interestingness_xrl.explainability.analysis.interaction.action_frequency import StateActionFrequencyAnalysis
from interestingness_xrl.scenarios.scenario_helper import ScenarioHelper
from interestingness_xrl.learning.agents import QValueBasedAgent
from interestingness_xrl.learning import get_features_from_index
from interestingness_xrl.util import print_line


class ContradictionAnalysis(AnalysisBase):
    """
    Represents an analysis of unexpected situations in which it was expected for the agent to behave in a certain
    manner, but the collected data informs us otherwise. Namely, it identifies contradictory-value states, i.e., states
    in which the actions' values distribution diverges from that of their rewards. Action probability distributions are
    derived from the rewards and values in the respective functions for that state. It also identifies
    contradictory-count states, i.e., states in which the actions' selection distribution (count) diverges from that of
    their values. Finally, it identifies contradictory goal states, i.e., states that were found to be sub-goals for the
    agent (local maxima) but were not in the known list of goal states (domain-knowledge).
    """

    def __init__(self, helper, agent, transition_value_analysis, state_action_freq_analysis,
                 min_state_count=5, action_dist_div_threshold=0.1):
        """
        Creates a new contradiction analysis.
        :param ScenarioHelper helper: the environment helper containing all necessary config and methods.
        :param QValueBasedAgent agent: the agent to be analyzed.
        :param TransitionValuesAnalysis transition_value_analysis: the analysis of the states' values.
        :param StateActionFrequencyAnalysis state_action_freq_analysis: the analysis of the (un)certain feature action pairs.
        :param int min_state_count: the minimum visits for a state to be considered contradictory.
        :param float action_dist_div_threshold: the threshold for the count, reward or value distribution divergence of
        actions for them to be considered different / non-aligned.
        """
        super().__init__(helper, agent)

        self.contradictory_value_states = []
        """ The states in which the actions' values distribution diverges from that of their rewards (s, n, jsd, [diff actions]). """

        self.contradictory_count_states = []
        """ The states in which the actions' selection distribution (count) diverges from that of their values (s, n, jsd, [diff actions]). """

        self.contradictory_goal_states = []
        """ The states that were found to be sub-goals for the agent (local maxima) but were not in the known list of goal states (s, n). """

        self.contradictory_feature_actions = []
        """ The state-feature - action pairs that were found to be certain but were not in the known list of associations (f, v, a). """

        self.state_rewards = {}
        self.state_counts = {}
        self.state_values = {}

        self.transition_value_analysis = transition_value_analysis
        self.state_action_freq_analysis = state_action_freq_analysis
        self.min_state_count = min_state_count
        self.action_dist_div_threshold = action_dist_div_threshold

    def difference_to(self, other):
        """
        Gets the difference between this analysis and another given analysis, i.e., the elements that are new and
        therefore missing in the other analysis. This can be used to highlight what is new in this analysis compared to
        one that was performed at a prior stage of the agent's behavior / learning.
        :param ContradictionAnalysis other: the other analysis to get the difference to.
        :return ContradictionAnalysis: an analysis resulting from the difference between this and the given analysis.
        """

        # initializes analysis diff
        diff_analysis = ContradictionAnalysis(
            self.helper, self.agent,
            self.transition_value_analysis, self.state_action_freq_analysis,
            self.min_state_count, self.action_dist_div_threshold)

        other_contradictory_value_states = set(s for s, *_ in other.contradictory_value_states)
        diff_analysis.contradictory_value_states = \
            [(s, *_) for s, *_ in self.contradictory_value_states
             if s not in other_contradictory_value_states]

        other_contradictory_count_states = set(s for s, *_ in other.contradictory_count_states)
        diff_analysis.contradictory_count_states = \
            [(s, *_) for s, *_ in self.contradictory_count_states
             if s not in other_contradictory_count_states]

        other_contradictory_goal_states = set(s for s, *_ in other.contradictory_goal_states)
        diff_analysis.contradictory_goal_states = \
            [(s, *_) for s, *_ in self.contradictory_goal_states
             if s not in other_contradictory_goal_states]

        diff_analysis.state_rewards = {}
        diff_analysis.state_counts = {}
        diff_analysis.state_values = {}
        for s, *_ in diff_analysis.contradictory_value_states:
            diff_analysis.state_rewards[s] = self.state_rewards[s]
            diff_analysis.state_values[s] = self.state_values[s]
        for s, *_ in diff_analysis.contradictory_count_states:
            diff_analysis.state_counts[s] = self.state_counts[s]
            diff_analysis.state_values[s] = self.state_values[s]

        diff_analysis.contradictory_feature_actions = list(
            set(self.contradictory_feature_actions) - set(other.contradictory_feature_actions))

        return diff_analysis

    def analyze(self):

        self.state_rewards = {}
        self.state_counts = {}
        self.state_values = {}

        # calculates all contradictory value states
        self.contradictory_value_states = []
        self.contradictory_count_states = []
        for s in range(self.agent.num_states):

            c_s = self.agent.c_s[s].item()

            # checks min support for all actions
            if any(self.agent.c_sa[s] < self.min_state_count):
                continue

            # compares to check contradictory value state (where value != reward action relative distribution)
            jsd, diff_actions = self._diff_action_dists(self.agent.q[s], self.agent.r_sa[s])
            if len(diff_actions) > 0:
                self.contradictory_value_states.append((s, c_s, jsd, diff_actions))
                self.state_rewards[s] = self.agent.r_sa[s].tolist()
                self.state_values[s] = self.agent.q[s].tolist()

            # compares to check contradictory count state (where count != value action relative distribution)
            jsd, diff_actions = self._diff_action_dists(self.agent.c_sa[s], self.agent.q[s])
            if len(diff_actions) > 0:
                self.contradictory_count_states.append((s, c_s, jsd, diff_actions))
                self.state_values[s] = self.agent.q[s].tolist()
                self.state_counts[s] = self.agent.c_sa[s].tolist()

        # gets all certain state-feature - action pairs
        certain_feat_actions = set()
        for f, v, _, max_actions in self.state_action_freq_analysis.certain_feats:
            for a in max_actions:
                certain_feat_actions.add((f, v, a))

        # checks for certain pairs that are not a known/given association
        self.contradictory_feature_actions = \
            list(certain_feat_actions - set(self.helper.get_known_feature_action_assocs()))

        # checks for local maxima that are not a given goal state
        local_maxima = set([x[0] for x in self.transition_value_analysis.local_maxima_states])
        contradictory_states = list(local_maxima - set(self.helper.get_known_goal_states()))
        # contradictory_states.update(goal_states - local_maxima)
        counts = self.agent.c_s[contradictory_states].tolist()
        self.contradictory_goal_states = list(zip(contradictory_states, counts))

        # sorts lists
        self.contradictory_value_states.sort(key=lambda e: -e[2])
        self.contradictory_count_states.sort(key=lambda e: -e[2])
        self.contradictory_feature_actions.sort()
        self.contradictory_goal_states.sort()

    def _diff_action_dists(self, table1, table2):
        """
        Checks whether the two given action tables for some state are very different/divergent.
        The tables contain values that can be converted into a probability distribution over actions.
        :param table1: the first action table.
        :param table2: the second action table.
        :rtype: tuple
        :return: a tuple with the overall JSD and the indexes in which the two action distributions diverge.
        """
        dist1 = self._normalize(table1)
        dist2 = self._normalize(table2)
        jsd = get_pairwise_jensen_shannon_divergence(dist1, dist2)
        jsd_sum = np.sum(jsd).item()
        return jsd_sum, np.where(jsd > self.action_dist_div_threshold / self.config.num_actions)[0].tolist()

    @staticmethod
    def _normalize(table):
        """
        Transforms the given table of values into a probability distribution in which indexes with higher values have
        a higher probability associated.
        :param table:
        :return ndarray: a probability distribution where the sum of the elements equals 1.
        """
        # normalizes each element between 0 and 1
        table = table.astype(np.float)
        t_max = np.max(table)
        t_min = np.min(table)
        t_range = t_max - t_min if t_max != t_min else 1.
        table = np.true_divide(table - t_min, t_range)

        # gets the exponential to get the probabilities
        probs = np.exp(table)

        # normalizes sum
        probs_sum = np.sum(probs)
        return np.true_divide(probs, probs_sum)

    def _save_report(self, file, write_console):
        feats_nbins = self.helper.get_features_bins()
        action_names = self.config.get_action_names()

        print_line('====================================', file, write_console)
        print_line('Action JS divergence threshold: {} per action:'.format(
            self.action_dist_div_threshold / self.config.num_actions), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} contradictory values states found):'.format(
            len(self.contradictory_value_states)), file, write_console)

        for s, n, jsd, diff_actions in self.contradictory_value_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec, True)
            action_labels = [action_names[a] for a in diff_actions]
            print_line('\t{}-{} (jsd: {:.3f}, count: {})'.format(s, feats_labels, jsd, n), file, write_console)
            print_line('\t\tDivergent actions: {}'.format(action_labels), file, write_console)
            self._print_actions('Rwd. dist', diff_actions, self.state_rewards[s], file, write_console)
            self._print_actions('Val. dist', diff_actions, self.state_values[s], file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} contradictory count states found):'.format(
            len(self.contradictory_count_states)), file, write_console)

        for s, n, jsd, diff_actions in self.contradictory_count_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec, True)
            action_labels = [action_names[a] for a in diff_actions]
            print_line('\t{}-{} (jsd: {:.3f}, count: {})'.format(s, feats_labels, jsd, n), file, write_console)
            print_line('\t\tDivergent actions: {}'.format(action_labels), file, write_console)
            self._print_actions('Val. dist', diff_actions, self.state_values[s], file, write_console)
            self._print_actions('Count dist', diff_actions, self.state_counts[s], file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} contradictory goal states found):'.format(
            len(self.contradictory_goal_states)), file, write_console)

        for s, n in self.contradictory_goal_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec, True)
            print_line('\t{}-{} (count: {})'.format(s, feats_labels, n), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} contradictory feature-action associations found):'.format(
            len(self.contradictory_feature_actions)), file, write_console)

        for f, v, a in self.contradictory_feature_actions:
            obs_vec = np.zeros(len(feats_nbins), np.uint32)
            obs_vec[f] = v
            feat_label = self.helper.get_features_labels(obs_vec)[f]
            print_line('\t{} - {}'.format(feat_label, action_names[a]), file, write_console)

    @staticmethod
    def _print_actions(title, diff_actions, table, file, write_console):
        print_line('\t\t{}:\t['.format(title), file, write_console, False)
        for a in range(len(table)):
            print_line(str(round(table[a], 3)).rjust(10), file, write_console, False)
            print_line('*' if a in diff_actions else ' ', file, write_console, False)
        print_line(']\n', file, write_console, False)

    def get_stats(self):
        return {
            'Num contradictory values states': (len(self.contradictory_value_states), 0., 1),
            'Num contradictory count states': (len(self.contradictory_count_states), 0., 1),
            'Num contradictory goal states': (len(self.contradictory_goal_states), 0., 1),
            'Num contradictory feature-action assocs': (len(self.contradictory_feature_actions), 0., 1),
        }

    def save_json(self, json_file_path):
        transition_value_analysis = self.transition_value_analysis
        self.transition_value_analysis = None

        state_action_freq_analysis = self.state_action_freq_analysis
        self.state_action_freq_analysis = None

        super().save_json(json_file_path)

        self.transition_value_analysis = transition_value_analysis
        self.state_action_freq_analysis = state_action_freq_analysis

    def _save_visual_report(self, path):

        feats_nbins = self.helper.get_features_bins()

        for s, *_ in self.contradictory_value_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'contradictory-value-s-{}.png'.format(s)))

        for s, *_ in self.contradictory_count_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'contradictory-count-s-{}.png'.format(s)))

        for s, *_ in self.contradictory_goal_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'contradictory-goal-s-{}.png'.format(s)))

    def get_sample_interesting_aspects(self, s, a, r, ns):

        feats_nbins = self.helper.get_features_bins()

        aspects = []
        for st, *_ in self.contradictory_value_states:
            if st == s:
                aspects.append('contradictory-value-s-{}'.format(s))
                break

        for st, *_ in self.contradictory_count_states:
            if st == s:
                aspects.append('contradictory-count-s-{}'.format(s))
                break

        for st, *_ in self.contradictory_goal_states:
            if st == s:
                aspects.append('contradictory-goal-s-{}'.format(s))
                break

        obs_vec = get_features_from_index(s, feats_nbins)
        for f, v, ac in self.contradictory_feature_actions:
            if obs_vec[f] == v and ac == a:
                aspects.append('contradictory-f-{}-v-{}-a-{}'.format(f, v, a))
                break

        return aspects

    def get_interestingness_names(self):
        return ['contradictory-value-s', 'contradictory-count-s', 'contradictory-goal-s', 'contradictory-f']
