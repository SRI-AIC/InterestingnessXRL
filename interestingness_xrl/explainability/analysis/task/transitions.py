__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
from os.path import join
from interestingness_xrl.explainability.analysis import AnalysisBase
from interestingness_xrl.explainability import get_distribution_evenness
from interestingness_xrl.scenarios.scenario_helper import ANY_FEATURE_IDX
from interestingness_xrl.learning import get_features_from_index
from interestingness_xrl.util import print_line


class TransitionAnalysis(AnalysisBase):
    """
    Represents an analysis of an agent's visits to next states after executing some action in some state. It computes
    the certain and uncertain state-action pairs, actions and state-feature values. Transition certainty is measured as
    the level of dispersion induced by executing some action in some state. Transitions leading to many different states
    have a high dispersion (information entropy) and are considered 'uncertain', while transitions leading to a few
    states have low dispersion and are considered 'certain'.
    """

    def __init__(self, helper, agent, min_state_count=5, trans_min_states=10, certain_trans_max_disp=0.1,
                 uncertain_trans_min_disp=0.9, certain_action_max_disp=0.1, uncertain_action_min_disp=0.6,
                 certain_feat_max_disp=0.1, uncertain_feat_min_disp=0.6):
        """
        Creates a new state-action transition analysis.
        :param ScenarioHelper helper: the environment helper containing all necessary config and methods.
        :param QValueBasedAgent agent: the agent to be analyzed.
        :param int min_state_count: the minimum ratio of state-action visits to state visits for a transition to be considered (un)certain.
        :param float trans_min_states: the minimum number of visited next states for a transition to be considered uncertain.
        :param float certain_trans_max_disp: the maximum dispersion of next-states for a transition to be considered (un)certain.
        :param float uncertain_trans_min_disp: the minimum dispersion of next-states for a transition to be considered uncertain.
        :param float certain_action_max_disp: the maximum average dispersion of next-states for an action to be considered certain.
        :param float uncertain_action_min_disp: the minimum average dispersion of next-states for an action to be considered uncertain.
        :param float certain_feat_max_disp: the maximum average dispersion of next-states for a feature-action to be considered certain.
        :param float uncertain_feat_min_disp: the minimum average dispersion of next-states for a feature-action to be considered uncertain.
        transitions for the feature value to be considerd (un)certain.
        """

        super().__init__(helper, agent)
        self.certain_trans = []
        """ The state-action transitions considered certain (s, a, supp, disp). """

        self.uncertain_trans = []
        """ The state-action transitions considered uncertain (s, a, supp, disp). """

        self.certain_actions = []
        """ The actions considered certain (action, disp). """

        self.uncertain_actions = []
        """ The actions considered uncertain (action, disp). """

        self.certain_feats = []
        """ The feature-action pairs considered certain (feat, val, a, disp). """

        self.uncertain_feats = []
        """ The feature-action pairs considered uncertain (feat, val, a, disp). """

        self.min_state_count = min_state_count
        self.trans_min_states = trans_min_states
        self.certain_trans_max_disp = certain_trans_max_disp
        self.uncertain_trans_min_disp = uncertain_trans_min_disp
        self.certain_action_max_disp = certain_action_max_disp
        self.uncertain_action_min_disp = uncertain_action_min_disp
        self.certain_feat_max_disp = certain_feat_max_disp
        self.uncertain_feat_min_disp = uncertain_feat_min_disp

    def difference_to(self, other):
        """
        Gets the difference between this analysis and another given analysis, i.e., the elements that are new and
        therefore missing in the other analysis. This can be used to highlight what is new in this analysis compared to
        one that was performed at a prior stage of the agent's behavior / learning.
        :param TransitionAnalysis other: the other analysis to get the difference to.
        :return TransitionAnalysis: an analysis resulting from the difference between this and the given analysis.
        """

        # initializes analysis diff
        diff_analysis = TransitionAnalysis(self.helper, self.agent, self.min_state_count, self.trans_min_states,
                                           self.certain_trans_max_disp, self.uncertain_trans_min_disp,
                                           self.certain_action_max_disp, self.uncertain_action_min_disp,
                                           self.certain_feat_max_disp, self.uncertain_feat_min_disp)

        other_certain_trans = set((s, a) for s, a, *_ in other.certain_trans)
        diff_analysis.certain_trans = \
            [(s, a, *_) for s, a, *_ in self.certain_trans
             if (s, a) not in other_certain_trans]

        other_uncertain_trans = set((s, a) for s, a, *_ in other.uncertain_trans)
        diff_analysis.uncertain_trans = \
            [(s, a, *_) for s, a, *_ in self.uncertain_trans
             if (s, a) not in other_uncertain_trans]

        other_certain_actions = set(a for a, *_ in other.certain_actions)
        diff_analysis.certain_actions = \
            [(a, *_) for a, *_ in self.certain_actions
             if a not in other_certain_actions]

        other_uncertain_actions = set(a for a, *_ in other.uncertain_actions)
        diff_analysis.uncertain_actions = \
            [(a, *_) for a, *_ in self.uncertain_actions
             if a not in other_uncertain_actions]

        other_certain_feats = set((f, v, a) for f, v, a, *_ in other.certain_feats)
        diff_analysis.certain_feats = \
            [(f, v, a, *_) for f, v, a, *_ in self.certain_feats
             if (f, v, a) not in other_certain_feats]

        other_uncertain_feats = set((f, v, a) for f, v, a, *_ in other.uncertain_feats)
        diff_analysis.uncertain_feats = \
            [(f, v, a, *_) for f, v, a, *_ in self.uncertain_feats
             if (f, v, a) not in other_uncertain_feats]

        return diff_analysis

    def analyze(self):

        self.certain_trans = []
        self.uncertain_trans = []

        # gets visited state-action pairs
        visited_s = np.nonzero(self.agent.c_s)[0]
        visited_sa_counts = self.agent.c_sa[visited_s]
        actions_disps = np.full((len(visited_s), self.config.num_actions), np.nan)

        # initializes feature-action transitions
        feats_nbins = self.helper.get_features_bins()
        num_feats = len(feats_nbins)
        feats_counts = [list()] * num_feats
        for f in range(num_feats):
            feats_counts[f] = np.zeros((feats_nbins[f], self.config.num_actions, self.config.num_states), np.uint)

        # calculates state-action dispersions
        for i in range(len(visited_s)):
            s = visited_s[i].item()
            obs_vec = get_features_from_index(s, feats_nbins)

            for a in range(self.config.num_actions):

                # ignore state-action visits with insufficient support
                state_action_supp = visited_sa_counts[i][a].item()
                if state_action_supp < self.min_state_count:
                    continue

                # updates feature-action transitions
                for f in range(len(feats_nbins)):
                    feats_counts[f][obs_vec[f]] += self.agent.c_sas[s]

                # calculates dispersion of possible transitions (only considers non-zero transitions)
                dist = self.agent.c_sas[s][a][np.nonzero(self.agent.c_sas[s][a])]
                disp = float(get_distribution_evenness(dist))
                actions_disps[i][a] = disp
                num_next_states = len(dist)

                # checks for certain and uncertain transitions
                if disp <= self.certain_trans_max_disp:
                    self.certain_trans.append((s, a, state_action_supp, disp))
                elif num_next_states >= self.trans_min_states and disp >= self.uncertain_trans_min_disp:
                    self.uncertain_trans.append((s, a, state_action_supp, disp))

        # analyzes un/certain actions
        self.certain_actions = []
        self.uncertain_actions = []
        for a in range(self.config.num_actions):

            action_disps = actions_disps[:, a]
            mean_action_disp = float(np.mean(action_disps[~np.isnan(action_disps)]))

            if mean_action_disp >= self.uncertain_action_min_disp:
                self.uncertain_actions.append((a, mean_action_disp))
            elif mean_action_disp <= self.certain_action_max_disp:
                self.certain_actions.append((a, mean_action_disp))

        # analyzes un/certain features
        self.certain_feats = []
        self.uncertain_feats = []
        for f in range(num_feats):
            for v in range(feats_nbins[f]):
                for a in range(self.config.num_actions):

                    # gets dispersion (only considers non-zero transitions)
                    dist = feats_counts[f][v][a][np.nonzero(feats_counts[f][v][a])]

                    # ignores all zero transitions
                    if len(dist) == 0:
                        continue
                    disp = float(get_distribution_evenness(dist))

                    if disp >= self.uncertain_feat_min_disp:
                        self.uncertain_feats.append((f, v, a, disp))
                    elif disp <= self.certain_feat_max_disp:
                        self.certain_feats.append((f, v, a, disp))

        # sorts lists
        self.certain_trans.sort(key=lambda e: e[3])
        self.uncertain_trans.sort(key=lambda e: -e[3])
        self.certain_actions.sort(key=lambda e: e[1])
        self.uncertain_actions.sort(key=lambda e: -e[1])
        self.certain_feats.sort(key=lambda e: e[3])
        self.uncertain_feats.sort(key=lambda e: -e[3])

    def _save_report(self, file, write_console):
        feats_nbins = self.helper.get_features_bins()
        action_names = self.config.get_action_names()

        print_line('Min. state-action support: {}'.format(self.min_state_count), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} certain state transitions found (max. dispersion: {}):'.format(
            len(self.certain_trans), self.certain_trans_max_disp), file, write_console)

        for s, a, supp, disp in self.certain_trans:
            feats_labels = self.helper.get_features_labels(get_features_from_index(s, feats_nbins), True)
            print_line('\t({}-{}, {}) (supp: {}, disp: {:.3f})'.format(
                s, feats_labels, action_names[a], supp, disp), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} uncertain state transitions found (min. dispersion: {}):'.format(
            len(self.uncertain_trans), self.uncertain_trans_min_disp), file, write_console)

        for s, a, supp, disp in self.uncertain_trans:
            feats_labels = self.helper.get_features_labels(get_features_from_index(s, feats_nbins), True)
            print_line('\t({}-{}, {}) (supp: {}, disp: {:.3f})'.format(
                s, feats_labels, action_names[a], supp, disp), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} certain actions found (max. dispersion: {}):'.format(
            len(self.certain_actions), self.certain_action_max_disp), file, write_console)

        for a, disp in self.certain_actions:
            print_line('\t{} (disp: {:.3f})'.format(action_names[a], disp), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} uncertain actions found (min. dispersion: {}):'.format(
            len(self.uncertain_actions), self.uncertain_action_min_disp), file, write_console)

        for a, disp in self.uncertain_actions:
            print_line('\t{} (disp: {:.3f})'.format(action_names[a], disp), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} certain state feature-action pairs found (max disp: {}):'.format(
            len(self.certain_feats), self.certain_feat_max_disp), file, write_console)

        num_feats = len(feats_nbins)
        for f, v, a, disp in self.certain_feats:
            obs_vec = np.zeros(num_feats, np.uint32)
            obs_vec[f] = v
            feat_label = self.helper.get_features_labels(obs_vec)[f]
            print_line('\t{} - {} (disp: {:.3f})'.format(feat_label, action_names[a], disp), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} uncertain state feature-action pairs found (min. disp: {}):'.format(
            len(self.uncertain_feats), self.uncertain_feat_min_disp), file, write_console)

        for f, v, a, disp in self.uncertain_feats:
            obs_vec = np.zeros(num_feats, np.uint32)
            obs_vec[f] = v
            feat_label = self.helper.get_features_labels(obs_vec)[f]
            print_line('\t{} - {} (disp: {:.3f})'.format(feat_label, action_names[a], disp), file, write_console)

    def get_stats(self):
        stats = {
            'Num certain state transitions': (len(self.certain_trans), 0., 1),
            'Num uncertain state transitions': (len(self.uncertain_trans), 0., 1),
            'Num certain actions': (len(self.certain_actions), 0., 1),
            'Num uncertain actions': (len(self.uncertain_actions), 0., 1),
            'Num certain state feature-action': (len(self.certain_feats), 0., 1),
            'Num uncertain state feature-action': (len(self.uncertain_feats), 0., 1),
        }
        return stats

    def _save_visual_report(self, path):

        feats_nbins = self.helper.get_features_bins()

        for s, a, *_ in self.certain_trans:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'certain-trans-s-{}-a-{}.png'.format(s, a)))

        for s, a, *_ in self.uncertain_trans:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'uncertain-trans-s-{}-a-{}.png'.format(s, a)))

        for f, v, a, *_ in self.certain_feats:
            obs_vec = [ANY_FEATURE_IDX] * len(feats_nbins)
            obs_vec[f] = v
            self.helper.save_features_image(obs_vec, join(path, 'certain-f-{}-v-{}-a-{}.png'.format(f, v, a)))

        for f, v, a, *_ in self.uncertain_feats:
            obs_vec = [ANY_FEATURE_IDX] * len(feats_nbins)
            obs_vec[f] = v
            self.helper.save_features_image(obs_vec, join(path, 'uncertain-f-{}-v-{}-a-{}.png'.format(f, v, a)))

    def get_sample_interesting_aspects(self, s, a, r, ns):

        feats_nbins = self.helper.get_features_bins()

        aspects = []
        for st, ac, *_ in self.certain_trans:
            if st == s and ac == a:
                aspects.append('certain-trans-s-{}-a-{}'.format(s, a))
                break

        for st, ac, *_ in self.uncertain_trans:
            if st == s and ac == a:
                aspects.append('uncertain-trans-s-{}-a-{}'.format(s, a))
                break

        obs_vec = get_features_from_index(s, feats_nbins)
        for f, v, ac, *_ in self.certain_feats:
            if obs_vec[f] == v and ac == a:
                aspects.append('certain-feats-f-{}-v-{}-a-{}'.format(f, v, a))
                break

        for f, v, ac, *_ in self.uncertain_feats:
            if obs_vec[f] == v and ac == a:
                aspects.append('uncertain-feats-f-{}-v-{}-a-{}'.format(f, v, a))
                break

        return aspects

    def get_interestingness_names(self):
        return ['certain-trans', 'uncertain-trans', 'certain-feats', 'uncertain-feats']
