__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
from os.path import join
from interestingness_xrl.scenarios.scenario_helper import ScenarioHelper
from interestingness_xrl.learning.agents import QValueBasedAgent
from interestingness_xrl.explainability.analysis import AnalysisBase
from interestingness_xrl.learning import get_features_from_index
from interestingness_xrl.util import print_line


class RecencyAnalysis(AnalysisBase):
    """
    Represents an analysis of an agent's recency with regards to visits to states and state-action pairs. Allows
    identifying which states and actions were visited earlier in the interaction with the environment but not recently.
    """

    def __init__(self, helper, agent, min_state_count=5, state_max_time_step=0.5, action_max_time_step=0.3):
        """
        Creates a new recency analysis.
        :param ScenarioHelper helper: the environment helper containing all necessary config and methods.
        :param QValueBasedAgent agent: the agent to be analyzed.
        :param int min_state_count: the minimum visits for states and state-action pairs to be considered early.
        :param float state_max_time_step: the maximum time-step visit (percentage) for a state to be considered early.
        :param float action_max_time_step: the maximum time-step visit (percentage) for a state-action pair to be considered early.
        """
        super().__init__(helper, agent)

        self.earlier_states = []
        """ The states considered to have been visited earlier (s, t, n). """

        self.earlier_actions = []
        """ The state-action pairs considered to have been visited earlier (s, a, t, n). """

        self.total_time_steps = 0
        """ The total number of time-steps in the agent's interaction with the environment. """

        self.min_state_count = min_state_count
        self.state_max_time_step = state_max_time_step
        self.action_max_time_step = action_max_time_step

    def difference_to(self, other):
        """
        Gets the difference between this analysis and another given analysis, i.e., the elements that are new and
        therefore missing in the other analysis. This can be used to highlight what is new in this analysis compared to
        one that was performed at a prior stage of the agent's behavior / learning.
        :param RecencyAnalysis other: the other analysis to get the difference to.
        :return RecencyAnalysis: an analysis resulting from the difference between this and the given analysis.
        """

        # initializes analysis diff
        diff_analysis = RecencyAnalysis(
            self.helper, self.agent, self.min_state_count, self.state_max_time_step, self.action_max_time_step)

        other_earlier_states = set(s for s, *_ in other.earlier_states)
        diff_analysis.earlier_states = \
            [(s, *_) for s, *_ in self.earlier_states
             if s not in other_earlier_states]

        other_earlier_actions = set((s, a) for s, a, *_ in other.earlier_actions)
        diff_analysis.earlier_actions = \
            [(s, a, *_) for s, a, *_ in self.earlier_actions
             if (s, a) not in other_earlier_actions]

        diff_analysis.total_time_steps = self.total_time_steps - other.total_time_steps

        return diff_analysis

    def analyze(self):

        # calculates total time-steps
        self.total_time_steps = np.max(self.agent.t_s).item()

        visited_s = np.where(np.logical_and(self.agent.c_s >= self.min_state_count, self.agent.t_s >= 0))
        visited_sa = np.where(np.logical_and(self.agent.c_sa >= self.min_state_count, self.agent.t_sa >= 0))

        # calculates recency of visited states and gets outliers
        visited_t_s = self.agent.t_s[visited_s]
        earlier_states_idxs = np.where(visited_t_s <= self.state_max_time_step * self.total_time_steps)[0]
        self.earlier_states = \
            list(zip(visited_s[0][earlier_states_idxs].tolist(),
                     visited_t_s[earlier_states_idxs].tolist(),
                     self.agent.c_s[visited_s][earlier_states_idxs].tolist()))

        # calculates recency of visited state-acton pairs and gets outliers
        visited_t_sa = self.agent.t_sa[visited_sa]
        earlier_sa_idxs = np.where(visited_t_sa <= self.action_max_time_step * self.total_time_steps)[0]
        self.earlier_actions = \
            list(zip(visited_sa[0][earlier_sa_idxs].tolist(),
                     visited_sa[1][earlier_sa_idxs].tolist(),
                     visited_t_sa[earlier_sa_idxs].tolist(),
                     self.agent.c_sa[visited_sa][earlier_sa_idxs].tolist()))

        # sorts lists
        self.earlier_states.sort(key=lambda e: e[1])
        self.earlier_actions.sort(key=lambda e: e[2])

    def _save_report(self, file, write_console):
        feats_nbins = self.helper.get_features_bins()
        action_names = self.config.get_action_names()

        print_line('====================================', file, write_console)
        print_line('{} total time-steps'.format(self.total_time_steps), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} earlier states found (min. support: {}, max time-step: {}):'.format(
            len(self.earlier_states), self.min_state_count, int(self.state_max_time_step * self.total_time_steps)),
            file, write_console)

        for s, t, n in self.earlier_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec, True)
            print_line('\t{}-{} (count: {}, last time-step: {})'.format(s, feats_labels, n, t), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} earlier state-action pairs found (min. support: {}, max time-step: {}):'.format(
            len(self.earlier_actions), self.min_state_count, int(self.action_max_time_step * self.total_time_steps)),
            file, write_console)

        for s, a, t, n in self.earlier_actions:
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec, True)
            print_line('\t{}-{} - {} (count: {}, last time-step: {})'.format(
                s, feats_labels, action_names[a], n, t), file, write_console)

    def get_stats(self):
        return {
            'Total time-steps': (self.total_time_steps, 0., 1),
            'Num earlier states': (len(self.earlier_states), 0., 1),
            'Num earlier state-actions': (len(self.earlier_actions), 0., 1)
        }

    def _save_visual_report(self, path):

        feats_nbins = self.helper.get_features_bins()

        for s, *_ in self.earlier_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'earlier-s-{}.png'.format(s)))

        for s, a, *_ in self.earlier_actions:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'earlier-s-{}-a-{}.png'.format(s, a)))

    def get_sample_interesting_aspects(self, s, a, r, ns):

        aspects = []
        for st, *_ in self.earlier_states:
            if st == s:
                aspects.append('earlier-s-{}'.format(s))
                break

        for st, ac, *_ in self.earlier_actions:
            if st == s and ac == a:
                aspects.append('earlier-ac-{}-{}'.format(s, a))
                break

        return aspects

    def get_interestingness_names(self):
        return ['earlier-s', 'earlier-ac']
