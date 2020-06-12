__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
from os.path import join
from pyfpgrowth.pyfpgrowth import FPTree
from interestingness_xrl.explainability.analysis import AnalysisBase
from interestingness_xrl.scenarios.scenario_helper import ScenarioHelper, ANY_FEATURE_IDX
from interestingness_xrl.learning.agents import QValueBasedAgent
from interestingness_xrl.learning import get_features_from_index
from interestingness_xrl.data_mining.jaccard_mining import find_patterns_above, generate_association_rules, \
    filter_maximal, find_patterns_below
from interestingness_xrl.explainability import get_distribution_evenness
from interestingness_xrl.util import print_line


class StateFrequencyAnalysis(AnalysisBase):
    """
    Represents an analysis of an agent's frequently-visited states, association patterns in the state features and
    association rules denoting consistent causality effects in the environment.
    """

    def __init__(self, helper, agent, min_state_count=100, max_state_count=1,
                 min_feat_set_count=10, min_feat_set_assoc=0.5, min_feat_rule_conf=0.5, max_feat_set_assoc=0.01):
        """
        Creates a new frequent states analysis.
        :param ScenarioHelper helper: the environment helper containing all necessary config and methods.
        :param QValueBasedAgent agent: the agent to be analyzed.
        :param int min_state_count: the minimal frequency of visits to a state for it to be considered frequent.
        :param int max_state_count: the maximal frequency of visits to a state for it to be considered infrequent.
        :param float min_feat_set_count: the minimal frequency of state features for a set to be considered frequent.
        :param float min_feat_set_assoc: the minimal association strength of state features for a set to be considered interesting.
        :param float min_feat_rule_conf: the minimal confidence of a feature rule antecedent for it to be considered interesting.
        :param float max_feat_set_assoc: the maximal association strength of state features for a set to be considered infrequent.
        """
        super().__init__(helper, agent)

        self.state_coverage = 0.
        """ The coverage of visits to the state space, i.e., the ratio between the visited and total states. """

        self.state_dispersion = 0.
        """ The dispersion of the distribution of the visits to the state space, i.e., how disperse the visits were. """

        self.total_count = 0
        """ The total number of visits / interaction time-steps with the environment. """

        self.freq_states = []
        """ The frequently-visited states (s, n). """

        self.infreq_states = []
        """ The infrequently-visited states (s, n). """

        self.freq_feature_sets = {}
        """ The association patterns in the state features (feat_set, jacc). """

        self.freq_feature_rules = []
        """ The association rules denoting consistent causality effects in the environment (ant, cons, freq, conf). """

        self.infreq_feature_sets = []
        """ The infrequent association patterns in the state features (feat_set). """

        self.min_state_count = min_state_count
        self.max_state_count = max_state_count
        self.min_feat_set_count = min_feat_set_count
        self.min_feat_set_assoc = min_feat_set_assoc
        self.min_feat_rule_conf = min_feat_rule_conf
        self.max_feat_set_assoc = max_feat_set_assoc

    def difference_to(self, other):
        """
        Gets the difference between this analysis and another given analysis, i.e., the elements that are new and
        therefore missing in the other analysis. This can be used to highlight what is new in this analysis compared to
        one that was performed at a prior stage of the agent's behavior / learning.
        :param StateFrequencyAnalysis other: the other analysis to get the difference to.
        :return StateFrequencyAnalysis: an analysis resulting from the difference between this and the given analysis.
        """

        # initializes analysis diff
        diff_analysis = StateFrequencyAnalysis(
            self.helper, self.agent, self.min_state_count, self.max_state_count, self.min_feat_set_count,
            self.min_feat_set_assoc, self.min_feat_rule_conf, self.max_feat_set_assoc)

        diff_analysis.state_coverage = self.state_coverage - other.state_coverage

        diff_analysis.state_dispersion = self.state_dispersion - other.state_dispersion

        diff_analysis.total_count = self.total_count - other.total_count

        other_freq_states = set(s for s, *_ in other.freq_states)
        diff_analysis.freq_states = \
            [(s, *_) for s, *_ in self.freq_states
             if s not in other_freq_states]

        other_infreq_states = set(s for s, *_ in other.infreq_states)
        diff_analysis.infreq_states = \
            [(s, *_) for s, *_ in self.infreq_states
             if s not in other_infreq_states]

        other_freq_feature_sets = set(feat_set for feat_set, *_ in other.freq_feature_sets)
        diff_analysis.freq_feature_sets = \
            [(feat_set, *_) for feat_set, *_ in self.freq_feature_sets
             if feat_set not in other_freq_feature_sets]

        other_freq_feature_rules = set((ant, cons) for ant, cons, *_ in other.freq_feature_rules)
        diff_analysis.freq_feature_rules = \
            [(ant, cons, *_) for ant, cons, *_ in self.freq_feature_rules
             if (ant, cons) not in other_freq_feature_rules]

        other_infreq_feature_sets = set(feat_set for feat_set, *_ in other.infreq_feature_sets)
        diff_analysis.infreq_feature_sets = \
            [(feat_set, *_) for feat_set, *_ in self.infreq_feature_sets
             if feat_set not in other_infreq_feature_sets]

        return diff_analysis

    def analyze(self):

        # gets total visited states and calculates coverage and dispersion
        visited_states_counts = self.agent.c_s[np.nonzero(self.agent.c_s)]
        self.state_coverage = float(len(visited_states_counts)) / self.config.num_states
        self.state_dispersion = float(get_distribution_evenness(visited_states_counts))

        feats_nbins = self.helper.get_features_bins()

        # builds states-as-transactions list and counts total state visits
        state_transactions = []
        self.total_count = 0
        for s in range(self.agent.num_states):
            cs = self.agent.c_s[s].item()
            self.total_count += cs

            # converts state idx to set of features and adds them as transactions
            obs_vec = get_features_from_index(s, feats_nbins)
            state_transaction = [(f, obs_vec[f].item()) for f in range(len(obs_vec))]
            for c in range(cs):
                state_transactions.append(state_transaction)

        # gets (in)frequent states
        self.freq_states = []
        self.infreq_states = []
        for s in range(self.agent.num_states):
            c_s = self.agent.c_s[s].item()
            if c_s >= self.min_state_count:
                self.freq_states.append((s, c_s))
            elif 0 < c_s <= self.max_state_count:
                self.infreq_states.append((s, c_s))

        # gets frequent feature-sets
        tree = FPTree(state_transactions, self.min_feat_set_count, None, None)
        patterns, no_patterns, counts = find_patterns_above(tree, self.min_feat_set_assoc)

        # filters out non-maximal patterns and 1-item patterns
        filter_maximal(patterns)
        self.freq_feature_sets = [(feat_set, jacc) for feat_set, jacc in patterns.items()]
        self.freq_feature_sets = [feat_set for feat_set in self.freq_feature_sets if len(feat_set[0]) > 1]

        # gets feature association rules
        self.freq_feature_rules = generate_association_rules(patterns, counts, self.min_feat_rule_conf)

        # gets infrequent feature-sets
        tree = FPTree(state_transactions, 1, None, None)
        self.infreq_feature_sets = find_patterns_below(tree, self.max_feat_set_assoc)
        self._filter_invalid_infreq_feat_sets()
        self._filter_maximal_infreq_feat_sets()

        # sorts lists
        self.freq_states.sort(key=lambda e: -e[1])
        self.infreq_states.sort(key=lambda e: e[1])
        self.freq_feature_sets.sort(key=lambda e: -e[1])
        self.freq_feature_rules.sort(key=lambda e: -e[3])
        self.infreq_feature_sets.sort(key=lambda e: len(e))

    def _filter_invalid_infreq_feat_sets(self):

        # filters out invalid feature-sets, i.e., feature values drawn from the same feature idx
        infreq_feature_sets = []
        for infreq_feature_set in self.infreq_feature_sets:
            idxs = set()
            valid = True
            for f, _ in infreq_feature_set:
                if f in idxs:
                    valid = False
                    break
                else:
                    idxs.add(f)
            if valid:
                infreq_feature_sets.append(infreq_feature_set)
        self.infreq_feature_sets = infreq_feature_sets

    def _filter_maximal_infreq_feat_sets(self):
        infreq_feature_sets = set(self.infreq_feature_sets)
        for super_set_lst in self.infreq_feature_sets:
            super_set = set(super_set_lst)
            for sub_set_lst in self.infreq_feature_sets:
                sub_set = set(sub_set_lst)
                if super_set != sub_set and sub_set_lst in infreq_feature_sets and sub_set.issubset(super_set):
                    infreq_feature_sets.remove(sub_set_lst)
        self.infreq_feature_sets = list(infreq_feature_sets)

    def _save_report(self, file, write_console):

        feats_nbins = self.helper.get_features_bins()

        print_line('====================================', file, write_console)
        print_line('Total states visited: {}'.format(self.total_count), file, write_console)
        print_line('Coverage of the state-space: {:.2f}%'.format(self.state_coverage * 100.), file, write_console)
        print_line('Dispersion of the visits to the state-space: {:.3f}'.format(
            self.state_dispersion), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} frequent states found (min. support: {}):'.format(
            len(self.freq_states), self.min_state_count), file, write_console)

        for s, n in self.freq_states:
            rf = n / float(self.total_count)
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec, True)
            print_line('\t{}-{} (count: {}, freq.: {:.3f})'.format(s, feats_labels, n, rf), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} infrequent states found (max. support: {}):'.format(
            len(self.infreq_states), self.max_state_count), file, write_console)

        for s, n in self.infreq_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            feats_labels = self.helper.get_features_labels(obs_vec, True)
            print_line('\t{}-{} (count: {})'.format(s, feats_labels, n), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} frequent feature-sets found (min. support: {}, min. assoc. strength: {}):'.format(
            len(self.freq_feature_sets), self.min_feat_set_count, self.min_feat_set_assoc), file, write_console)

        num_feats = len(feats_nbins)
        for feat_set, jacc in self.freq_feature_sets:
            feats_labels = self._get_feats_labels(feat_set, num_feats)
            print_line('\t{} (jacc: {:.3f})'.format(feats_labels, jacc), file, write_console)

        print_line('====================================', file, write_console)
        print_line('{} interesting feature-rules found (min. confidence: {}):'.format(
            len(self.freq_feature_rules), self.min_feat_rule_conf), file, write_console)

        for ant, cons, n, conf in self.freq_feature_rules:
            n /= float(self.total_count)
            ant = self._get_feats_labels(ant, num_feats)
            cons = self._get_feats_labels(cons, num_feats)
            print_line('\t{} => {} (freq: {:.3f}, conf: ({:.3f}))'.format(ant, cons, n, conf), file,
                       write_console)

        print_line('====================================', file, write_console)
        print_line('{} infrequent feature-sets found (max. assoc. strength: {}):'.format(
            len(self.infreq_feature_sets), self.max_feat_set_assoc), file, write_console)

        for feat_set in self.infreq_feature_sets:
            feats_labels = self._get_feats_labels(feat_set, num_feats)
            print_line('\t{}'.format(feats_labels), file, write_console)

    def _get_feats_labels(self, feat_set, num_feats):
        obs_vec = np.zeros(num_feats, np.uint32)
        for f, v in feat_set:
            obs_vec[f] = v
        feats_labels = self.helper.get_features_labels(obs_vec)
        feats_labels = [feats_labels[f] for f, _ in feat_set]
        return feats_labels

    def get_stats(self):
        return {
            'Num total visits': (self.total_count, 0., 1),
            'State coverage': (self.state_coverage, 0., 1),
            'Dispersion of visits': (self.state_dispersion, 0., 1),
            'Num frequent states': (len(self.freq_states), 0., 1),
            'Num infrequent states': (len(self.infreq_states), 0., 1),
            'Num frequent feature-sets': (len(self.freq_feature_sets), 0., 1),
            'Num feature-rules': (len(self.freq_feature_rules), 0., 1),
            'Num infrequent feature-sets': (len(self.infreq_feature_sets), 0., 1),
        }

    def _save_visual_report(self, path):

        feats_nbins = self.helper.get_features_bins()

        for s, *_ in self.freq_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'freq-s-{}.png'.format(s)))

        for s, *_ in self.infreq_states:
            obs_vec = get_features_from_index(s, feats_nbins)
            self.helper.save_features_image(obs_vec, join(path, 'infreq-s-{}.png'.format(s)))

        for feat_set, *_ in self.freq_feature_sets:
            obs_vec = [ANY_FEATURE_IDX] * len(feats_nbins)
            set_name = ''
            for f, v in feat_set:
                obs_vec[f] = v
                set_name += '-{}-{}'.format(f, v)
            self.helper.save_features_image(obs_vec, join(path, 'freq-fv{}.png'.format(set_name)))

        for feat_set in self.infreq_feature_sets:
            obs_vec = [ANY_FEATURE_IDX] * len(feats_nbins)
            set_name = ''
            for f, v in feat_set:
                obs_vec[f] = v
                set_name += '-{}-{}'.format(f, v)
            self.helper.save_features_image(obs_vec, join(path, 'infreq-fv{}.png'.format(set_name)))

    def get_sample_interesting_aspects(self, s, a, r, ns):

        feats_nbins = self.helper.get_features_bins()

        aspects = []
        for st, *_ in self.freq_states:
            if st == s:
                aspects.append('frequent-s-{}'.format(s))
                break

        for st, *_ in self.infreq_states:
            if st == s:
                aspects.append('infrequent-s-{}'.format(s))
                break

        obs_vec = get_features_from_index(s, feats_nbins)
        for feat_set, *_ in self.freq_feature_sets:
            if all(obs_vec[f] == v for f, v in feat_set):
                aspects.append('frequent-feature-set-s-{}'.format(s))
                break

        for feat_set in self.infreq_feature_sets:
            if all(obs_vec[f] == v for f, v in feat_set):
                aspects.append('infrequent-feature-set-s-{}'.format(s))
                break

        return aspects

    def get_interestingness_names(self):
        return ['frequent-s', 'infrequent-s', 'frequent-feature-set', 'infrequent-feature-set']
