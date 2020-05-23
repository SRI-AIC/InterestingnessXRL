__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
from os.path import join
from interestingness_xrl.explainability.analysis import AnalysisBase
from interestingness_xrl.scenarios.scenario_helper import ScenarioHelper, ANY_FEATURE_IDX
from interestingness_xrl.learning.agents import QValueBasedAgent
from interestingness_xrl.learning import get_features_from_index
from interestingness_xrl.explainability import get_distribution_evenness, get_outliers_dist_mean, get_diff_means
from interestingness_xrl.util import print_line


class StateActionFrequencyAnalysis(AnalysisBase):
    """
    Represents an analysis of the agent's history of action selection/execution with the environment. Namely, it
    calculates the state-action coverage, the mean dispersion of action executions, and the (un)certain states and state
    features. Certainty of a state or state feature is measured as how uneven the action selection in that state or in
    the presence of that feature is.
    """

    def __init__(self, helper, agent, min_state_count=5,
                 certain_state_max_disp=0.1, uncertain_state_min_disp=0.9,
                 certain_feat_max_disp=0.1, uncertain_feat_min_disp=0.9):
        """
        Creates a new state-acton frequency analysis.
        :param ScenarioHelper helper: the environment helper containing all necessary config and methods.
        :param QValueBasedAgent agent: the agent to be analyzed.
        :param int min_state_count: the minimum ratio of state visits for a state to be considered either certain or uncertain.
        :param float certain_state_max_disp: the maximum dispersion of action executions for a state to be considered certain.
        :param float uncertain_state_min_disp: the minimum dispersion of action executions for a state to be considered uncertain.
        :param float certain_feat_max_disp: the maximum dispersion of action executions for a state-feature to be considered certain.
        :param float uncertain_feat_min_disp: the minimum dispersion of action executions for a state-feature to be considered uncertain.
        """
        super().__init__(helper, agent)

        self.state_action_coverage = 0.
        """ The coverage of visits to the state space, i.e., the ratio between the visited and total states. """

        self.mean_action_dispersion = (0., 0., 0)
        """ The dispersion of the distribution of the visits to the state space, i.e., how disperse the visits were (mean, std, n). """

        self.certain_states = []
        """ The states where action selection is considered certain (s, disp, max_actions). """

        self.uncertain_states = []
        """ The states where action selection is considered uncertain (s, disp). """

        self.certain_feats = []
        """ The state-features considered certain (feat, val, disp, max_action). """

        self.uncertain_feats = []
        """ The state-features considered uncertain (feat, val, disp). """

        self.min_state_count = min_state_count
        self.certain_state_max_disp = certain_state_max_disp
        self.uncertain_state_min_disp = uncertain_state_min_disp
        self.certain_feat_max_disp = certain_feat_max_disp
        self.uncertain_feat_min_disp = uncertain_feat_min_disp

    def difference_to(self, other):
        """
        Gets the difference between this analysis and another given analysis, i.e., the elements that are new and
        therefore missing in the other analysis. This can be used to highlight what is new in this analysis compared to
        one that was performed at a prior stage of the agent's behavior / learning.
        :param StateActionFrequencyAnalysis other: the other analysis to get the difference to.
        :return StateActionFrequencyAnalysis: an analysis resulting from the difference between this and the given analysis.
        """

        # initializes analysis diff
        diff_analysis = StateActionFrequencyAnalysis(
            self.helper, self.agent,
            self.min_state_count, self.certain_state_max_disp, self.uncertain_state_min_disp,
            self.certain_feat_max_disp, self.uncertain_feat_min_disp)

        diff_analysis.state_action_coverage = self.state_action_coverage - other.state_action_coverage

        mean1, std1, n1 = self.mean_action_dispersion
        mean2, std2, n2 = other.mean_action_dispersion
        diff_analysis.avg_reward = get_diff_means(mean1, std1, n1, mean2, std2, n2)

        other_certain_states = set(s for s, *_ in other.certain_states)
        diff_analysis.uncertain_actions = \
            [(s, *_) for s, *_ in self.certain_states
             if s not in other_certain_states]

        other_uncertain_states = set(s for s, *_ in other.uncertain_states)
        diff_analysis.uncertain_states = \
            [(s, *_) for s, *_ in self.uncertain_states
             if s not in other_uncertain_states]

        other_certain_feats = set((f, v) for f, v, *_ in other.certain_feats)
        diff_analysis.certain_feats = \
            [(f, v, *_) for f, v, *_ in self.certain_feats
             if (f, v) not in other_certain_feats]

        other_uncertain_feats = set((f, v) for f, v, *_ in other.uncertain_feats)
        diff_analysis.uncertain_feats = \
            [(f, v, *_) for f, v, *_ in self.uncertain_feats
             if (f, v) not in other_uncertain_feats]

        return diff_analysis

    def analyze(self):
        """
        Performs an analysis of the behavior of the agent during its interaction with the environment.
        Namely, it calculates the state-action coverage, the mean dispersion of action executions, and the (un)certain
        states and state features.
        :return:
        """

        # gets visited state-action pairs
        visited_s = np.nonzero(self.agent.c_s)[0]
        visited_sa_counts = self.agent.c_sa[visited_s]

        # calculates coverage
        num_visited_sa = np.count_nonzero(visited_sa_counts)
        total_sa = len(visited_sa_counts) * self.config.num_actions
        self.state_action_coverage = float(num_visited_sa) / total_sa

        # calculates mean action execution dispersion
        state_dispersions = [get_distribution_evenness(c_a) for c_a in visited_sa_counts]
        self.mean_action_dispersion = (
            np.mean(state_dispersions).item(),
            np.std(state_dispersions).item(),
            len(state_dispersions))

        # analyzes un/certain states
        self.certain_states = []
        self.uncertain_states = []
        for i in range(len(visited_s)):

            # check state visit ratio
            s = visited_s[i].item()
            if self.agent.c_s[s] < self.min_state_count:
                continue

            # checks dispersions
            disp = state_dispersions[i]
            if disp >= self.uncertain_state_min_disp:
                self.uncertain_states.append((s, float(disp)))
            elif disp <= self.certain_state_max_disp:
                # if certain state, get the action(s) usually selected
                max_actions = get_outliers_dist_mean(visited_sa_counts[i], 1.5, below=False)
                self.certain_states.append((s, float(disp), max_actions))

        # gets action counts for each feature
        feats_nbins = self.helper.get_features_bins()
        num_feats = len(feats_nbins)
        feats_counts = [list()] * num_feats
        for f in range(num_feats):
            feats_counts[f] = np.zeros((feats_nbins[f], self.config.num_actions), np.uint)
        for i in range(len(visited_sa_counts)):
            c_sa = visited_sa_counts[i]
            s = visited_s[i]
            obs_vec = get_features_from_index(s, feats_nbins)
            for f in range(len(feats_nbins)):
                feats_counts[f][obs_vec[f]] += c_sa

        # analyzes un/certain state-features
        self.certain_feats = []
        self.uncertain_feats = []
        for f in range(num_feats):
            for v in range(feats_nbins[f]):

                # checks dispersions
                disp = float(get_distribution_evenness(feats_counts[f][v]))
                if disp >= self.uncertain_feat_min_disp:
                    self.uncertain_feats.append((f, v, disp))
                elif disp <= self.certain_feat_max_disp:
                    # if certain feature, get the action(s) usually selected
                    max_actions = get_outliers_dist_mean(feats_counts[f][v], 1.5, below=False)
                    self.certain_feats.append((f, v, disp, max_actions))

        # sorts lists
        self.certain_states.sort(key=lambda e: e[1])
        self.uncertain_states.sort(key=lambda e: -e[1])
        self.certain_feats.sort(key=lambda e: e[2])
        self.uncertain_feats.sort(key=lambda e: -e[2])

    def _save_report(self, file, write_console):
        feats_nbins = self.helper.get_features_bins()
        action_names = self.config.get_action_names()

        print_line('====================================', file, write_console)
        print_line('Coverage of the state-action space (for visited states): {:.2f}%'.format(
            self.state_action_coverage * 100.), file, write_console)

        mean, std, n = self.mean_action_dispersion
        print_line('Mean dispersion of the execution of actions in visited states: {:.3f} ± {:.3f} (count: {})'
                   .format(mean, std, n), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} certain states found (max. dispersion: {}):'.format(
            len(self.certain_states), self.certain_state_max_disp), file, write_console)

        for s, disp, max_actions in self.certain_states:
            max_action_labels = [action_names[a] for a in max_actions]
            feats_labels = self.helper.get_features_labels(get_features_from_index(s, feats_nbins), True)
            print_line('\t{}-{} (mean disp.: {:.3f}, max actions {})'.format(
                s, feats_labels, disp, max_action_labels), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} uncertain states found (min. dispersion: {}):'.format(
            len(self.uncertain_states), self.uncertain_state_min_disp), file, write_console)

        for s, disp in self.uncertain_states:
            feats_labels = self.helper.get_features_labels(get_features_from_index(s, feats_nbins), True)
            print_line('\t{}-{} (mean disp.: {:.3f})'.format(s, feats_labels, disp), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} certain state features found (max. dispersion: {}):'.format(
            len(self.certain_feats), self.certain_feat_max_disp), file, write_console)

        num_feats = len(feats_nbins)
        for f, v, disp, max_actions in self.certain_feats:
            max_action_labels = [action_names[a] for a in max_actions]
            obs_vec = np.zeros(num_feats, np.uint32)
            obs_vec[f] = v
            feat_label = self.helper.get_features_labels(obs_vec)[f]
            print_line('\t{} (mean disp.: {:.3f}, max actions: {})'.format(
                feat_label, disp, max_action_labels), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} uncertain state features found (min. dispersion: {}):'.format(
            len(self.uncertain_feats), self.uncertain_feat_min_disp), file, write_console)

        for f, v, disp in self.uncertain_feats:
            obs_vec = np.zeros(num_feats, np.uint32)
            obs_vec[f] = v
            feat_label = self.helper.get_features_labels(obs_vec)[f]
            print_line('\t{} (mean disp.: {:.3f})'.format(feat_label, disp), file, write_console)

    def get_stats(self):
        return {
            'State-action coverage': (self.state_action_coverage, 0., 1),
            'Mean dispersion action-execution': self.mean_action_dispersion,
            'Num certain states': (len(self.certain_states), 0., 1),
            'Num uncertain states': (len(self.uncertain_states), 0., 1),
            'Num certain state features': (len(self.certain_feats), 0., 1),
            'Num uncertain state features': (len(self.uncertain_feats), 0., 1),
        }

    def _save_visual_report(self, path):

        feats_nbins = self.helper.get_features_bins()

        for s, *_ in self.certain_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'certain-exec-s-{}.png'.format(s)))

        for s, *_ in self.uncertain_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'uncertain-exec-s-{}.png'.format(s)))

        for f, v, *_ in self.certain_feats:
            obs_vec = [ANY_FEATURE_IDX] * len(feats_nbins)
            obs_vec[f] = v
            self.helper.save_features_image(obs_vec, join(path, 'certain-exec-f-{}-v-{}.png'.format(f, v)))

        for f, v, *_ in self.uncertain_feats:
            obs_vec = [ANY_FEATURE_IDX] * len(feats_nbins)
            obs_vec[f] = v
            self.helper.save_features_image(obs_vec, join(path, 'uncertain-exec-f-{}-v-{}.png'.format(f, v)))

    def get_sample_interesting_aspects(self, s, a, r, ns):

        feats_nbins = self.helper.get_features_bins()

        aspects = []
        for st, *_ in self.certain_states:
            if st == s:
                aspects.append('certain-exec-s-{}'.format(s))
                break

        for st, *_ in self.uncertain_states:
            if st == s:
                aspects.append('uncertain-exec-s-{}'.format(s))
                break

        obs_vec = get_features_from_index(s, feats_nbins)
        for f, v, *_ in self.certain_feats:
            if obs_vec[f] == v:
                aspects.append('certain-exec-f-{}-v-{}'.format(f, v))
                break

        for f, v, *_ in self.uncertain_feats:
            if obs_vec[f] == v:
                aspects.append('uncertain-exec-f-{}-v-{}'.format(f, v))
                break

        return aspects

    def get_interestingness_names(self):
        return ['certain-exec-s', 'uncertain-exec-s', 'certain-exec-f', 'uncertain-exec-f']
