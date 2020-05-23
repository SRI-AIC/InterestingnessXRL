__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
from os.path import join
from interestingness_xrl.explainability import get_outliers_dist_mean, get_diff_means
from interestingness_xrl.explainability.analysis import AnalysisBase
from interestingness_xrl.scenarios.scenario_helper import ScenarioHelper, ANY_FEATURE_IDX
from interestingness_xrl.learning.agents import QValueBasedAgent
from interestingness_xrl.learning import get_features_from_index
from interestingness_xrl.util import print_line


class RewardAnalysis(AnalysisBase):
    """
    Represents an analysis of an agent's (modeled) reward function. It extracts information on the states that are, on
    average among all actions, significantly more or less rewarding than others (outliers). It also calculates the
    average reward received per action, and identifies the (state) feature-action pairs that are, on average among all
    visited states, significantly more or less rewarding than others (outliers).
    """

    def __init__(self, helper, agent, min_state_count=5, state_outlier_stds=2, feature_outlier_stds=2):
        """
        Creates a new reward analysis.
        :param ScenarioHelper helper: the environment helper containing all necessary config and methods.
        :param QValueBasedAgent agent: the agent to be analyzed.
        :param int min_state_count: the minimum visits for a state-action pair to be considered an outlier.
        :param float state_outlier_stds: the threshold for the mean reward MAD z-score of a state for it to be considered an outlier.
        :param float feature_outlier_stds: the threshold for the mean reward MAD z-score of a feature-action pair for it to be considered an outlier.
        """
        super().__init__(helper, agent)

        self.state_action_outliers = []
        """ The state-action pairs considered as outliers with regards to their mean reward (s, a, mean, n). """

        self.action_rwds_avg = []
        """ The average reward values for each action (mean, std_dev). """

        self.feature_action_outliers = []
        """ The feature-action pairs considered as outliers with regards to their mean reward (f, v, a, mean, std_dev, n). """

        self.avg_reward = (0., 0., 0)
        """ The overall average reward collected by the agent (mean, std, n)."""

        self.min_state_count = min_state_count
        self.state_outlier_stds = state_outlier_stds
        self.feature_outlier_stds = feature_outlier_stds

    def difference_to(self, other):
        """
        Gets the difference between this analysis and another given analysis, i.e., the elements that are new and
        therefore missing in the other analysis. This can be used to highlight what is new in this analysis compared to
        one that was performed at a prior stage of the agent's behavior / learning.
        :param RewardAnalysis other: the other analysis to get the difference to.
        :return RewardAnalysis: an analysis resulting from the difference between this and the given analysis.
        """

        # initializes analysis diff
        diff_analysis = RewardAnalysis(self.helper, self.agent, self.min_state_count, self.state_outlier_stds,
                                       self.feature_outlier_stds)

        other_state_action_outliers = set((s, a) for s, a, *_ in other.state_action_outliers)
        diff_analysis.state_action_outliers = \
            [(s, a, *_) for s, a, *_ in self.state_action_outliers
             if (s, a) not in other_state_action_outliers]

        other_feature_action_outliers = set((f, v, a) for f, v, a, *_ in other.feature_action_outliers)
        diff_analysis.feature_action_outliers = \
            [(f, v, a, *_) for f, v, a, *_ in self.feature_action_outliers
             if (f, v, a) not in other_feature_action_outliers]

        diff_analysis.action_rwds_avg = []
        for a in range(len(self.action_rwds_avg)):
            mean1, std1, n1 = self.action_rwds_avg[a]
            mean2, std2, n2 = other.action_rwds_avg[a]
            diff_analysis.action_rwds_avg.append(get_diff_means(mean1, std1, n1, mean2, std2, n2))

        mean1, std1, n1 = self.avg_reward
        mean2, std2, n2 = other.avg_reward
        diff_analysis.avg_reward = get_diff_means(mean1, std1, n1, mean2, std2, n2)

        return diff_analysis

    def analyze(self):

        # gets visited state-action pairs with sufficient support
        visited_sa = np.where(self.agent.c_sa >= self.min_state_count)
        visited_r_sa = self.agent.r_sa[visited_sa]

        # gets state-action pairs with outlier rewards
        state_action_outliers = get_outliers_dist_mean(visited_r_sa, self.state_outlier_stds)
        s_idxs = visited_sa[0][state_action_outliers].tolist()
        a_idxs = visited_sa[1][state_action_outliers].tolist()
        rwds = visited_r_sa[state_action_outliers].tolist()
        counts = self.agent.c_sa[visited_sa][state_action_outliers].tolist()
        self.state_action_outliers = list(zip(s_idxs, a_idxs, rwds, counts))

        # gets average reward per action
        action_rwd_means = [float(np.mean(visited_r_sa[visited_sa[1] == a])) for a in range(self.config.num_actions)]
        action_rwd_stds = [float(np.std(visited_r_sa[visited_sa[1] == a])) for a in range(self.config.num_actions)]
        action_rwd_ns = [len(visited_r_sa[visited_sa[1] == a]) for a in range(self.config.num_actions)]
        self.action_rwds_avg = list(zip(action_rwd_means, action_rwd_stds, action_rwd_ns))

        # collects all rewards per state feature and action
        feats_nbins = self.helper.get_features_bins()
        feats_rwds = [[] for _ in range(len(feats_nbins))]
        for f in range(len(feats_nbins)):
            feats_rwds[f] = [[] for _ in range(feats_nbins[f])]
            for v in range(feats_nbins[f]):
                feats_rwds[f][v] = [[] for _ in range(self.config.num_actions)]
        for s, a, rwd in zip(visited_sa[0], visited_sa[1], visited_r_sa):

            # gets features for each visited state
            obs_vec = get_features_from_index(s, feats_nbins)

            # for each feature, add reward to the corresponding feature value bucket
            for f in range(len(obs_vec)):
                feats_rwds[f][obs_vec[f]][a].append(rwd)

        # gets average reward per state feature and extracts outliers
        feats_rwds_avg = []
        for f in range(len(feats_nbins)):
            for v in range(feats_nbins[f]):
                for a in range(self.config.num_actions):
                    rwds = feats_rwds[f][v][a]
                    if len(rwds) == 0:
                        continue
                    feats_rwds_avg.append(
                        (f, v, a, float(np.mean(rwds)), float(np.std(rwds)), len(rwds)))
        fva_outliers_idxs = get_outliers_dist_mean([x[3] for x in feats_rwds_avg], self.feature_outlier_stds)
        self.feature_action_outliers = [feats_rwds_avg[i] for i in fva_outliers_idxs]

        # gets overall average reward received
        self.avg_reward = (float(np.mean(visited_r_sa)), float(np.std(visited_r_sa)), len(visited_r_sa))

        # sorts lists
        self.state_action_outliers.sort(key=lambda e: -e[2])
        self.feature_action_outliers.sort(key=lambda e: -e[3])

    def _save_report(self, file, write_console):

        feats_nbins = self.helper.get_features_bins()
        action_names = self.config.get_action_names()

        rwd_avg, rwd_std, n = self.avg_reward
        print_line('====================================', file, write_console)
        print_line('Average overall reward: {:.3f} ± {:.3f} (count: {})'.format(rwd_avg, rwd_std, n),
                   file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} state-action outliers found (outlier threshold: {}):'.format(
            len(self.state_action_outliers), self.state_outlier_stds), file, write_console)

        for s, a, rwd, n in self.state_action_outliers:
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec)
            print_line('\t{}-{} - {} (mean reward: {:.3f}, count: {})'.format(
                s, feats_labels, action_names[a], rwd, n), file, write_console)

        print_line('====================================', file, write_console)
        print_line('Actions\' average rewards:', file, write_console)

        for a in range(len(self.action_rwds_avg)):
            rwd_avg, rwd_std, n = self.action_rwds_avg[a]
            print_line('\t{}: {:.3f} ± {:.3f} (count: {})'.format(
                action_names[a], rwd_avg, rwd_std, n), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} feature-action outliers found (outlier threshold: {}):'.format(
            len(self.feature_action_outliers), self.feature_outlier_stds), file, write_console)

        for f, v, a, rwd_avg, rwd_std, n in self.feature_action_outliers:
            obs_vec = np.zeros(len(feats_nbins), np.uint32)
            obs_vec[f] = v
            feat_label = self.helper.get_features_labels(obs_vec)[f]
            print_line('\t{} - {} (avg. reward: {:.3f} ± {:.3f}, count: {})'.format(
                feat_label, action_names[a], rwd_avg, rwd_std, n), file, write_console)

    def get_stats(self):
        stats = {
            'Mean overall reward': self.avg_reward,
            'Num state-action outliers': (len(self.state_action_outliers), 0., 1),
            'Num feature-action outliers': (len(self.feature_action_outliers), 0., 1)
        }
        action_names = self.config.get_action_names()
        for a in range(len(self.action_rwds_avg)):
            rwd_avg, rwd_std, n = self.action_rwds_avg[a]
            stats['Action {} mean reward'.format(action_names[a])] = (rwd_avg, rwd_std, n)
        return stats

    def _save_visual_report(self, path):

        feats_nbins = self.helper.get_features_bins()

        for s, a, *_ in self.state_action_outliers:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'mean-reward-outlier-s-{}-a-{}.png'.format(s, a)))

        for f, v, a, *_ in self.feature_action_outliers:
            obs_vec = [ANY_FEATURE_IDX] * len(feats_nbins)
            obs_vec[f] = v
            self.helper.save_features_image(
                obs_vec, join(path, 'mean-reward-outlier-f-{}-v-{}-a-{}.png'.format(f, v, a)))

    def get_sample_interesting_aspects(self, s, a, r, ns):

        feats_nbins = self.helper.get_features_bins()

        aspects = []
        for st, ac, *_ in self.state_action_outliers:
            if st == s and ac == a:
                aspects.append('mean-reward-outlier-s-{}-a-{}'.format(s, a))
                break

        obs_vec = get_features_from_index(s, feats_nbins)
        for f, v, ac, *_ in self.feature_action_outliers:
            if obs_vec[f] == v and ac == a:
                aspects.append('mean-reward-outlier-f-{}-v-{}-a-{}'.format(f, v, a))
                break

        return aspects

    def get_interestingness_names(self):
        return ['mean-reward-outlier-s', 'mean-reward-outlier-f']
