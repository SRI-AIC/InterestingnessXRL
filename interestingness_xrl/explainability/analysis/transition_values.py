__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
from os.path import join
from interestingness_xrl.explainability import get_outliers_dist_mean, group_by_key
from interestingness_xrl.explainability.analysis import AnalysisBase
from interestingness_xrl.scenarios.scenario_helper import ScenarioHelper
from interestingness_xrl.learning.agents import QValueBasedAgent
from interestingness_xrl.learning import get_features_from_index
from interestingness_xrl.util import print_line


class TransitionValuesAnalysis(AnalysisBase):
    """
    Represents an analysis of an agent's state value function with regards to state transitions. It calculates the local
    and absolute minima and maxima states, i.e., the states whose values are <=/>= than all possible next states,
    respectively. From this it also calculates the maximal difference states.
    """

    def __init__(self, helper, agent,
                 min_state_count=5, min_transition_count=5, state_diff_var_outlier_stds=2):
        """
        Creates a new value analysis.
        :param ScenarioHelper helper: the environment helper containing all necessary config and methods.
        :param QValueBasedAgent agent: the agent to be analyzed.
        :param int min_state_count: the minimum visits for a state-action pair to be considered an outlier.
        :param float min_transition_count: the minimum visits for a transition to be considered for state minima/maxima calculation.
        :param float state_diff_var_outlier_stds: the threshold for the value difference variance of a state for it to be considered an outlier.
        """
        super().__init__(helper, agent)

        self.local_minima_states = []
        """ The local minima states, i.e., states whose values are <= than possible next states (s, val, next_val_avg, n) """

        self.local_maxima_states = []
        """ The local maxima states, i.e., states whose values are >= than possible next states (s, val, next_val_avg, n) """

        self.absolute_minima_states = []
        """ The absolute minima states, i.e., states whose values are <= than all other states (s, val, next_val_avg, n) """

        self.absolute_maxima_states = []
        """ The absolute maxima states, i.e., states whose values are >= than all other states (s, val, next_val_avg, n) """

        self.val_diff_mean_action_outliers = []
        """ The state-action pairs considered as outliers with regards to difference in value to a possible next state (s, a, diff_mean, nsa) """

        self.val_diff_variance_state_outliers = []
        """ The states considered as outliers with regards to the variance of the difference in value to possible next states (s, val_diff_var, n) """

        self.min_state_count = min_state_count
        self.min_transition_count = min_transition_count
        self.state_diff_outlier_stds = state_diff_var_outlier_stds

    def difference_to(self, other):
        """
        Gets the difference between this analysis and another given analysis, i.e., the elements that are new and
        therefore missing in the other analysis. This can be used to highlight what is new in this analysis compared to
        one that was performed at a prior stage of the agent's behavior / learning.
        :param TransitionValuesAnalysis other: the other analysis to get the difference to.
        :return TransitionValuesAnalysis: an analysis resulting from the difference between this and the given analysis.
        """

        # initializes analysis diff
        diff_analysis = TransitionValuesAnalysis(
            self.helper, self.agent,
            self.min_state_count, self.min_transition_count, self.state_diff_outlier_stds)

        other_local_minima_states = set(s for s, *_ in other.local_minima_states)
        diff_analysis.local_minima_states = \
            [(s, *_) for s, *_ in self.local_minima_states
             if s not in other_local_minima_states]

        other_local_maxima_states = set(s for s, *_ in other.local_maxima_states)
        diff_analysis.local_maxima_states = \
            [(s, *_) for s, *_ in self.local_maxima_states
             if s not in other_local_maxima_states]

        other_absolute_minima_states = set(s for s, *_ in other.absolute_minima_states)
        diff_analysis.absolute_minima_states = \
            [(s, *_) for s, *_ in self.absolute_minima_states
             if s not in other_absolute_minima_states]

        other_absolute_maxima_states = set(s for s, *_ in other.absolute_maxima_states)
        diff_analysis.absolute_maxima_states = \
            [(s, *_) for s, *_ in self.absolute_maxima_states
             if s not in other_absolute_maxima_states]

        other_val_diff_mean_action_outliers = set((s, a) for s, a, *_ in other.val_diff_mean_action_outliers)
        diff_analysis.val_diff_mean_action_outliers = \
            [(s, a, *_) for s, a, *_ in self.val_diff_mean_action_outliers
             if (s, a) not in other_val_diff_mean_action_outliers]

        other_val_diff_variance_state_outliers = set(s for s, *_ in other.val_diff_variance_state_outliers)
        diff_analysis.val_diff_variance_state_outliers = \
            [(s, *_) for s, *_ in self.val_diff_variance_state_outliers
             if s not in other_val_diff_variance_state_outliers]

        return diff_analysis

    def analyze(self):

        # gets visited state-action pairs with sufficient support
        visited_sa = np.where(self.agent.c_sa >= self.min_state_count)

        # gets states' values (max q)
        visited_q_sa = self.agent.q[visited_sa]
        visited_q_s = group_by_key(list(zip(visited_sa[0].tolist(), visited_q_sa)))
        visited_s = np.array([s for s, _ in visited_q_s])
        visited_v_s = np.array([np.max(values).item() for _, values in visited_q_s])

        # gets local minima and maxima states
        self.local_minima_states = []
        self.local_maxima_states = []
        self.absolute_minima_states = []
        self.absolute_maxima_states = []
        self.val_diff_mean_action_outliers = []
        val_diff_variances = []
        absolute_min_val = np.finfo(float).max
        absolute_max_val = np.finfo(float).min
        next_val_diff_avgs = np.zeros(self.agent.c_sa.shape)
        for i, s in enumerate(visited_s):
            s = s.item()
            v_s = visited_v_s[i].item()
            c_s = 0
            num_actions = 0

            local_minimum = True
            local_maximum = True
            next_val_avg_minimums = []
            next_val_avg_maximums = []
            next_val_diff_vars = []

            for a in range(self.agent.num_actions):

                # gets visited next states using this action (at least one transition observed)
                visited_next_s = np.where(self.agent.c_sas[s][a] > 0)[0]
                if len(visited_next_s) == 0:
                    continue

                # gets effective state-action count and updates effective state count
                c_sa = np.sum(self.agent.c_sas[s][a][ns] for ns in visited_next_s).item()
                c_s += c_sa
                num_actions += 1

                # checks whether all next states have a value (max s-a value)
                # lower than or equal to the current state (possible local minimum)
                if local_minimum and np.all([v_s <= np.max(self.agent.q[ns]) for ns in visited_next_s]):
                    next_val_avg_minimums.append(
                        (c_sa,
                         np.sum(
                             [np.max(self.agent.q[ns]) * (self.agent.c_sas[s][a][ns] / float(c_sa))
                              for ns in visited_next_s])))
                else:
                    local_minimum = False

                # checks whether all next states have a value (max s-a value)
                # higher than or equal to the current state (possible local maximum)
                if local_maximum and np.all([v_s >= np.max(self.agent.q[ns]) for ns in visited_next_s]):
                    next_val_avg_maximums.append(
                        (c_sa,
                         np.sum(
                             [np.max(self.agent.q[ns]) * (self.agent.c_sas[s][a][ns] / float(c_sa))
                              for ns in visited_next_s])))
                else:
                    local_maximum = False

                # appends mean and variance of difference to next states' values
                next_val_diff_vars.append((c_sa, np.var([np.max(self.agent.q[ns]) - v_s for ns in visited_next_s])))
                next_val_diff_avgs[s][a] = np.mean([abs(np.max(self.agent.q[ns]) - v_s) for ns in visited_next_s])

            # continue only if effective state count is higher than 0 (does have transitions)
            if c_s == 0:
                continue

            # tests if state was found to be a local minimum or maximum and gets weighted avg of next state values
            # only if agent tested all actions a state can be a minimum/maximum
            if local_minimum and num_actions == self.agent.num_actions:
                next_val_avg = np.sum(ns_val * (c_sa / float(c_s)) for c_sa, ns_val in next_val_avg_minimums).item()
                self.local_minima_states.append((s, v_s, next_val_avg, c_s))

                # tests for absolute minimum
                if v_s <= absolute_min_val:
                    if v_s < absolute_min_val:
                        absolute_min_val = v_s
                        self.absolute_minima_states = []
                    self.absolute_minima_states.append((s, v_s, next_val_avg, c_s))

            if local_maximum and num_actions == self.agent.num_actions:
                next_val_avg = np.sum(ns_val * (c_sa / float(c_s)) for c_sa, ns_val in next_val_avg_maximums).item()
                self.local_maxima_states.append((s, v_s, next_val_avg, c_s))

                # tests for absolute maximum
                if v_s >= absolute_max_val:
                    if v_s > absolute_max_val:
                        absolute_max_val = v_s
                        self.absolute_maxima_states = []
                    self.absolute_maxima_states.append((s, v_s, next_val_avg, c_s))

            # calculates weighted average of value difference variance
            next_val_diff_var = np.sum(
                ns_abs_diff * (c_sa / float(c_s)) for c_sa, ns_abs_diff in next_val_diff_vars).item()
            val_diff_variances.append((s, next_val_diff_var, c_s))

        # calculates value difference variance state outliers
        state_outliers_idxs = get_outliers_dist_mean([x[1] for x in val_diff_variances], self.state_diff_outlier_stds)
        self.val_diff_variance_state_outliers = [val_diff_variances[i] for i in state_outliers_idxs]

        # calculates value difference mean state-action outliers (above mean)
        visited_next_val_diff_avgs_sa = np.nonzero(next_val_diff_avgs)
        visited_next_val_diff_avgs = next_val_diff_avgs[visited_next_val_diff_avgs_sa]
        outlier_idxs = get_outliers_dist_mean(visited_next_val_diff_avgs, self.state_diff_outlier_stds, below=False)
        s_idxs = visited_next_val_diff_avgs_sa[0][outlier_idxs].tolist()
        a_idxs = visited_next_val_diff_avgs_sa[1][outlier_idxs].tolist()
        diff_avgs = visited_next_val_diff_avgs[outlier_idxs].tolist()
        sa_counts = self.agent.c_sa[visited_next_val_diff_avgs_sa].tolist()
        self.val_diff_mean_action_outliers = list(zip(s_idxs, a_idxs, diff_avgs, sa_counts))

        # sorts lists
        self.local_minima_states.sort(key=lambda e: e[1])
        self.local_maxima_states.sort(key=lambda e: -e[1])
        self.absolute_minima_states.sort(key=lambda e: e[1])
        self.absolute_maxima_states.sort(key=lambda e: -e[1])
        self.val_diff_mean_action_outliers.sort(key=lambda e: -e[2])
        self.val_diff_variance_state_outliers.sort(key=lambda e: -e[1])

    def _save_report(self, file, write_console):

        feats_nbins = self.helper.get_features_bins()

        print_line('====================================', file, write_console)
        print_line('{} local minima states found (min. transition support: {}):'.format(
            len(self.local_minima_states), self.min_transition_count), file, write_console)

        for s, val, ns_val_avg, n in self.local_minima_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec, True)
            print_line('\t{}-{} (count: {}, value: {:.3f} <= avg next values: {:.3f})'.format(
                s, feats_labels, n, val, ns_val_avg), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} absolute minima states found (min. transition support: {}):'.format(
            len(self.absolute_minima_states), self.min_transition_count), file, write_console)

        for s, val, ns_val_avg, n in self.absolute_minima_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec, True)
            print_line('\t{}-{} (count: {}, value: {:.3f} <= avg next values: {:.3f})'.format(
                s, feats_labels, n, val, ns_val_avg), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} local maxima states found (min. transition support: {}):'.format(
            len(self.local_maxima_states), self.min_transition_count), file, write_console)

        for s, val, ns_val_avg, n in self.local_maxima_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec, True)
            print_line('\t{}-{} (count: {}, value: {:.3f} >= avg next values: {:.3f})'.format(
                s, feats_labels, n, val, ns_val_avg), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} absolute maxima states found (min. transition support: {}):'.format(
            len(self.absolute_maxima_states), self.min_transition_count), file, write_console)

        for s, val, ns_val_avg, n in self.absolute_maxima_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec, True)
            print_line('\t{}-{} (count: {}, value: {:.3f} >= avg next values: {:.3f})'.format(
                s, feats_labels, n, val, ns_val_avg), file, write_console)

        print_line('====================================', file, write_console)
        print_line(
            '{} value difference mean state outliers found (min. transition support: {}, outlier stds: {}):'.format(
                len(self.val_diff_mean_action_outliers), self.min_transition_count, self.state_diff_outlier_stds),
            file, write_console)

        action_names = self.config.get_action_names()
        for s, a, diff, n in self.val_diff_mean_action_outliers:
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec, True)
            print_line('\t{}-{} - {} (trans. count: {}, mean value diff.: {:.3f})'.format(
                s, feats_labels, action_names[a], n, diff), file, write_console)

        print_line('====================================', file, write_console)
        print_line(
            '{} value difference variance state-action outliers found (min. transition support: {}, outlier stds: {}):'.
                format(len(self.val_diff_variance_state_outliers), self.min_transition_count,
                       self.state_diff_outlier_stds), file, write_console)

        for s, diff, n in self.val_diff_variance_state_outliers:
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec, True)
            print_line('\t{}-{} (count: {}, value diff. variance: {:.3f})'.format(
                s, feats_labels, n, diff), file, write_console)

    def get_stats(self):
        return {
            'Num local minima': (len(self.local_minima_states), 0., 1),
            'Num absolute minima': (len(self.absolute_minima_states), 0., 1),
            'Num local maxima': (len(self.local_maxima_states), 0., 1),
            'Num absolute maxima': (len(self.absolute_maxima_states), 0., 1),
            'Num val diff mean state outliers': (len(self.val_diff_mean_action_outliers), 0., 1),
            'Num val diff variance state-action outliers': (len(self.val_diff_variance_state_outliers), 0., 1),
        }

    def _save_visual_report(self, path, clear=True):

        feats_nbins = self.helper.get_features_bins()

        for s, *_ in self.local_minima_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'local-minimum-s-{}.png'.format(s)))

        for s, *_ in self.absolute_minima_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'absolute-minimum-s-{}.png'.format(s)))

        for s, *_ in self.local_maxima_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'local-maximum-s-{}.png'.format(s)))

        for s, *_ in self.absolute_maxima_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'absolute-maximum-s-{}.png'.format(s)))

        for s, a, *_ in self.val_diff_mean_action_outliers:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'val-diff-mean-outlier-s-{}-a-{}.png'.format(s, a)))

        for s, *_ in self.val_diff_variance_state_outliers:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'val-diff-variance-outlier-s-{}.png'.format(s)))

    def get_sample_interesting_aspects(self, s, a, r, ns):

        aspects = []
        for st, *_ in self.local_minima_states:
            if st == s:
                aspects.append('local-minimum-s-{}'.format(s))
                break

        for st, *_ in self.absolute_minima_states:
            if st == s:
                aspects.append('absolute-minimum-s-{}'.format(s))
                break

        for st, *_ in self.local_maxima_states:
            if st == s:
                aspects.append('local-maximum-s-{}'.format(s))
                break

        for st, *_ in self.absolute_maxima_states:
            if st == s:
                aspects.append('absolute-maximum-s-{}'.format(s))
                break

        for st, ac, *_ in self.val_diff_mean_action_outliers:
            if st == s and ac == a:
                aspects.append('val-diff-mean-outlier-s-{}-a-{}'.format(s, a))
                break

        for st, *_ in self.val_diff_variance_state_outliers:
            if st == s:
                aspects.append('val-diff-variance-outlier-s-{}'.format(s))
                break

        return aspects

    def get_interestingness_names(self):
        return ['local-minimum-s', 'absolute-minimum-s', 'local-maximum-s', 'absolute-maximum-s',
                'val-diff-mean-outlier-s', 'val-diff-variance-outlier-s']
