__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
from os.path import join
from interestingness_xrl.scenarios.scenario_helper import ScenarioHelper, ANY_FEATURE_IDX
from interestingness_xrl.explainability import get_outliers_dist_mean, group_by_key, get_diff_means
from interestingness_xrl.explainability.analysis import AnalysisBase
from interestingness_xrl.learning.agents import QValueBasedAgent
from interestingness_xrl.learning import get_features_from_index
from interestingness_xrl.util import print_line


class ValueAnalysis(AnalysisBase):
    """
    Represents an analysis of an agent's state and action value functions. It extracts information on the states that
    are significantly more or less valued than others (outliers). It also calculates the average value for each action
    and identifies the (state) features that are, on average among all visited states and actions, significantly more or
    less rewarding than others (outliers).
    """

    def __init__(self, helper, agent,
                 min_state_count=5, state_outlier_stds=2, pred_error_outlier_stds=2, feature_outlier_stds=2):
        """
        Creates a new value analysis.
        :param ScenarioHelper helper: the environment helper containing all necessary config and methods.
        :param QValueBasedAgent agent: the agent to be analyzed.
        :param int min_state_count: the minimum visits for a state-action pair to be considered an outlier.
        :param float state_outlier_stds: the threshold for the value of a state for it to be considered an outlier.
        :param float pred_error_outlier_stds: the threshold for the value of a state for it to be considered an outlier.
        :param float feature_outlier_stds: the threshold for the mean value of a feature for it to be considered an outlier.
        """
        super().__init__(helper, agent)

        self.mean_val_state_action_outliers = []
        """ The state-action pairs considered as outliers with regards to their mean value (s, [max actions], val, n). """

        self.pred_error_state_action_outliers = []
        """ The state-action pairs considered as outliers with regards to their mean prediction error (s, a, dq, n). """

        self.action_vals_avg = []
        """ The average reward values for each action (mean, std_dev, n). """

        self.mean_val_feature_outliers = []
        """ The feature-action pairs considered as outliers with regards to their mean value (f, v, a, mean, std_dev). """

        self.pred_error_feature_outliers = []
        """ The feature-action pairs considered as outliers with regards to their mean prediction error (f, v, a, mean, std_dev). """

        self.avg_value = (0., 0., 0)
        """ The overall average value of states (mean, std, n)."""

        self.avg_pred_error = (0., 0., 0)
        """ The average prediction error among all states and actions (mean, std, n)."""

        self.min_state_count = min_state_count
        self.state_value_outlier_stds = state_outlier_stds
        self.pred_error_outlier_stds = pred_error_outlier_stds
        self.feature_outlier_stds = feature_outlier_stds

    def difference_to(self, other):
        """
        Gets the difference between this analysis and another given analysis, i.e., the elements that are new and
        therefore missing in the other analysis. This can be used to highlight what is new in this analysis compared to
        one that was performed at a prior stage of the agent's behavior / learning.
        :param ValueAnalysis other: the other analysis to get the difference to.
        :return ValueAnalysis: an analysis resulting from the difference between this and the given analysis.
        """

        # initializes analysis diff
        diff_analysis = ValueAnalysis(
            self.helper, self.agent,
            self.min_state_count, self.state_value_outlier_stds,
            self.pred_error_outlier_stds, self.feature_outlier_stds)

        other_mean_val_state_action_outliers = set(s for s, *_ in other.mean_val_state_action_outliers)
        diff_analysis.mean_val_state_action_outliers = \
            [(s, *_) for s, *_ in self.mean_val_state_action_outliers
             if s not in other_mean_val_state_action_outliers]

        other_pred_error_state_action_outliers = set((s, a) for s, a, *_ in other.pred_error_state_action_outliers)
        diff_analysis.pred_error_state_action_outliers = \
            [(s, a, *_) for s, a, *_ in self.pred_error_state_action_outliers
             if (s, a) not in other_pred_error_state_action_outliers]

        diff_analysis.action_rwds_avg = []
        for a in range(len(self.action_vals_avg)):
            mean1, std1, n1 = self.action_vals_avg[a]
            mean2, std2, n2 = other.action_vals_avg[a]
            diff_analysis.action_rwds_avg.append(get_diff_means(mean1, std1, n1, mean2, std2, n2))

        other_mean_val_feature_outliers = set((f, v) for f, v, *_ in other.mean_val_feature_outliers)
        diff_analysis.mean_val_feature_outliers = \
            [(f, v, *_) for f, v, *_ in self.mean_val_feature_outliers
             if (f, v) not in other_mean_val_feature_outliers]

        other_pred_error_feature_outliers = set((f, v, a) for f, v, a, *_ in other.pred_error_feature_outliers)
        diff_analysis.pred_error_feature_outliers = \
            [(f, v, a, *_) for f, v, a, *_ in self.pred_error_feature_outliers
             if (f, v, a) not in other_pred_error_feature_outliers]

        mean1, std1, n1 = self.avg_value
        mean2, std2, n2 = other.avg_value
        diff_analysis.avg_value = get_diff_means(mean1, std1, n1, mean2, std2, n2)

        mean1, std1, n1 = self.avg_pred_error
        mean2, std2, n2 = other.avg_pred_error
        diff_analysis.avg_pred_error = get_diff_means(mean1, std1, n1, mean2, std2, n2)

        return diff_analysis

    def analyze(self):

        # gets visited state-action pairs with sufficient support
        visited_sa = np.where(self.agent.c_sa >= self.min_state_count)

        # gets overall average state value
        visited_q_sa = self.agent.q[visited_sa]
        self.avg_value = (float(np.mean(visited_q_sa)), float(np.std(visited_q_sa)), len(visited_q_sa))

        # gets states' values (max q)
        visited_q_s = group_by_key(list(zip(visited_sa[0].tolist(), visited_q_sa)))
        visited_s = np.array([s for s, _ in visited_q_s])
        visited_v_s = np.array([np.mean(values).item() for _, values in visited_q_s])

        # gets states with outlier maximum values and the corresponding actions
        state_outliers = get_outliers_dist_mean(visited_v_s, self.state_value_outlier_stds)
        s_idxs = visited_s[state_outliers].tolist()
        as_idxs = [np.where(self.agent.q[visited_s[i]] == visited_v_s[i])[0].tolist() for i in state_outliers]
        s_values = visited_v_s[state_outliers].tolist()
        counts = self.agent.c_s[s_idxs].tolist()
        self.mean_val_state_action_outliers = list(zip(s_idxs, as_idxs, s_values, counts))

        # gets overall average prediction error
        visited_dq_sa = self.agent.dq[visited_sa]
        self.avg_pred_error = (float(np.mean(visited_dq_sa)), float(np.std(visited_dq_sa)), len(visited_dq_sa))

        # gets state-action pairs with outlier mean prediction errors (delta-q)
        state_action_outliers = get_outliers_dist_mean(visited_dq_sa, self.pred_error_outlier_stds)
        s_idxs = visited_sa[0][state_action_outliers].tolist()
        a_idxs = visited_sa[1][state_action_outliers].tolist()
        pred_errors = visited_dq_sa[state_action_outliers].tolist()
        counts = self.agent.c_sa[visited_sa][state_action_outliers].tolist()
        self.pred_error_state_action_outliers = list(zip(s_idxs, a_idxs, pred_errors, counts))

        # gets average value per action
        action_values = group_by_key(list(zip(visited_sa[1].tolist(), visited_q_sa)))
        self.action_vals_avg = [(a, np.mean(values).item(), np.std(values).item()) for a, values in action_values]

        # collects maximal values for all state features
        feats_nbins = self.helper.get_features_bins()
        feats_vals = [[] for _ in range(len(feats_nbins))]
        feats_dqs = [[] for _ in range(len(feats_nbins))]
        for f in range(len(feats_nbins)):
            feats_vals[f] = [[] for _ in range(feats_nbins[f])]
            feats_dqs[f] = [[] for _ in range(feats_nbins[f])]
            for v in range(feats_nbins[f]):
                feats_dqs[f][v] = [[] for _ in range(self.config.num_actions)]

        for i, val in enumerate(visited_v_s):

            # gets features for each visited state
            s = visited_s[i]
            obs_vec = get_features_from_index(s, feats_nbins)

            # for each feature, adds the value to the corresponding feature value bucket
            for f in range(len(obs_vec)):
                feats_vals[f][obs_vec[f]].append(val)

        for i, dq in enumerate(visited_dq_sa):
            # gets features for each visited state-action pair
            s = visited_sa[0][i]
            a = visited_sa[1][i]
            obs_vec = get_features_from_index(s, feats_nbins)

            # for each feature, adds the value to the corresponding feature value bucket
            for f in range(len(obs_vec)):
                feats_dqs[f][obs_vec[f]][a].append(dq)

        # gets average value and pred. error per state feature and extracts outliers
        feats_vals_avg = []
        feats_pred_error_avg = []
        for f in range(len(feats_nbins)):
            for v in range(feats_nbins[f]):
                q_values = feats_vals[f][v]
                if len(q_values) > 0:
                    feats_vals_avg.append((f, v, float(np.mean(q_values)), float(np.std(q_values))))
                for a in range(self.config.num_actions):
                    dqs = feats_dqs[f][v][a]
                    if len(dqs) > 0:
                        feats_pred_error_avg.append((f, v, a, float(np.mean(dqs)), float(np.std(dqs))))

        feat_outliers_idxs = get_outliers_dist_mean([x[2] for x in feats_vals_avg], self.feature_outlier_stds)
        self.mean_val_feature_outliers = [feats_vals_avg[i] for i in feat_outliers_idxs]

        feat_outliers_idxs = get_outliers_dist_mean([x[3] for x in feats_pred_error_avg], self.feature_outlier_stds)
        self.pred_error_feature_outliers = [feats_pred_error_avg[i] for i in feat_outliers_idxs]

        # sorts lists
        self.mean_val_state_action_outliers.sort(key=lambda e: -e[2])
        self.pred_error_state_action_outliers.sort(key=lambda e: -e[2])
        self.mean_val_feature_outliers.sort(key=lambda e: -e[2])
        self.pred_error_feature_outliers.sort(key=lambda e: -e[3])

    def _save_report(self, file, write_console):

        feats_nbins = self.helper.get_features_bins()
        action_names = self.config.get_action_names()

        mean, std, n = self.avg_value
        print_line('====================================', file, write_console)
        print_line('Average overall value: {:.3f} ± {:.3f} (count: {})'
                   .format(mean, std, n), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} state-action value outliers found (min. support: {}, outlier threshold: {}):'.format(
            len(self.mean_val_state_action_outliers),
            self.min_state_count, self.state_value_outlier_stds), file, write_console)

        for s, actions, val, n in self.mean_val_state_action_outliers:
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec, True)
            act_labels = [action_names[a] for a in actions]
            print_line('\t{}-{} (value: {:.3f}, count: {})\n\t\tMax actions: {}'.format(
                s, feats_labels, val, n, act_labels), file, write_console)

        mean, std, n = self.avg_pred_error
        print_line('====================================', file, write_console)
        print_line('Average overall prediction error: {:.3f} ± {:.3f} (count: {})'.format(
            mean, std, n), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} state-action prediction error outliers found (min. support: {}, outlier threshold: {}):'
                   .format(len(self.pred_error_state_action_outliers), self.min_state_count,
                           self.pred_error_outlier_stds), file, write_console)

        for s, a, pred_error, n in self.pred_error_state_action_outliers:
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec, True)
            print_line('\t{}-{} - {} (avg. prediction error: {:.3f}, count: {})'.format(
                s, feats_labels, action_names[a], pred_error, n), file, write_console)

        print_line('====================================', file, write_console)
        print_line('Actions\' average values:', file, write_console)

        for a, avg, std in self.action_vals_avg:
            print_line('\t{}: {:.3f} ± {:.3f}'.format(action_names[a], avg, std), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} feature value outliers found (outlier threshold: {}):'.format(
            len(self.mean_val_feature_outliers), self.feature_outlier_stds), file, write_console)

        for f, v, avg, std in self.mean_val_feature_outliers:
            obs_vec = np.zeros(len(feats_nbins), np.uint32)
            obs_vec[f] = v
            feat_label = self.helper.get_features_labels(obs_vec)[f]
            print_line('\t{} (avg. value: {:.3f} ± {:.3f})'.format(feat_label, avg, std), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} feature pred. error outliers found (outlier threshold: {}):'.format(
            len(self.pred_error_feature_outliers), self.feature_outlier_stds), file, write_console)

        for f, v, a, avg, std in self.pred_error_feature_outliers:
            obs_vec = np.zeros(len(feats_nbins), np.uint32)
            obs_vec[f] = v
            feat_label = self.helper.get_features_labels(obs_vec)[f]
            print_line('\t{} - {} (avg. pred. error: {:.3f} ± {:.3f})'.format(
                feat_label, action_names[a], avg, std), file, write_console)

    def get_stats(self):
        stats = {
            'Mean value': self.avg_value,
            'Mean prediction error': self.avg_pred_error,
            'Num state-action value outliers': (len(self.mean_val_state_action_outliers), 0., 1),
            'Num state-action pred error outliers': (len(self.pred_error_state_action_outliers), 0., 1),
            'Num feature value outliers': (len(self.mean_val_feature_outliers), 0., 1),
            'Num feature pred error outliers': (len(self.pred_error_feature_outliers), 0., 1),
        }
        action_names = self.config.get_action_names()
        for a in range(len(self.action_vals_avg)):
            rwd_avg, rwd_std, n = self.action_vals_avg[a]
            stats['Action {} mean value'.format(action_names[a])] = (rwd_avg, rwd_std, n)
        return stats

    def _save_visual_report(self, path):

        feats_nbins = self.helper.get_features_bins()

        for s, *_ in self.mean_val_state_action_outliers:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'mean-val-outlier-s-{}.png'.format(s)))

        for s, a, *_ in self.pred_error_state_action_outliers:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'pred-error-outlier-s-{}-a-{}.png'.format(s, a)))

        for f, v, *_ in self.mean_val_feature_outliers:
            obs_vec = [ANY_FEATURE_IDX] * len(feats_nbins)
            obs_vec[f] = v
            self.helper.save_features_image(obs_vec, join(path, 'mean-val-outlier-f-{}-v-{}.png'.format(f, v)))

        for f, v, a, *_ in self.pred_error_feature_outliers:
            obs_vec = [ANY_FEATURE_IDX] * len(feats_nbins)
            obs_vec[f] = v
            self.helper.save_features_image(
                obs_vec, join(path, 'pred-error-outlier-f-{}-v-{}-a-{}.png'.format(f, v, a)))

    def get_sample_interesting_aspects(self, s, a, r, ns):

        feats_nbins = self.helper.get_features_bins()

        aspects = []
        for st, *_ in self.mean_val_state_action_outliers:
            if st == s:
                aspects.append('mean-val-outlier-s-{}'.format(s))
                break

        for st, a, *_ in self.pred_error_state_action_outliers:
            if st == s:
                aspects.append('pred-error-outlier-s-{}-a-{}'.format(s, a))
                break

        obs_vec = get_features_from_index(s, feats_nbins)
        for f, v, *_ in self.mean_val_feature_outliers:
            if obs_vec[f] == v:
                aspects.append('mean-val-outlier-f-{}-v-{}'.format(f, v))
                break

        for f, v, ac, *_ in self.pred_error_feature_outliers:
            if obs_vec[f] == v and a == ac:
                aspects.append('pred-error-outlier-f-{}-v-{}-a-{}'.format(f, v, a))
                break

        return aspects

    def get_interestingness_names(self):
        return ['mean-val-outlier-s', 'pred-error-outlier-s', 'mean-val-outlier-f', 'pred-error-outlier-f']
