__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

from interestingness_xrl.scenarios.scenario_helper import ScenarioHelper
from interestingness_xrl.learning.agents import QValueBasedAgent
from interestingness_xrl.explainability.analysis import AnalysisBase
from interestingness_xrl.explainability.analysis.task.transitions import TransitionAnalysis
from interestingness_xrl.explainability.analysis.task.rewards import RewardAnalysis
from interestingness_xrl.explainability.analysis.interaction.state_frequency import StateFrequencyAnalysis
from interestingness_xrl.explainability.analysis.interaction.action_frequency import StateActionFrequencyAnalysis
from interestingness_xrl.explainability.analysis.interaction.values import ValueAnalysis
from interestingness_xrl.explainability.analysis.interaction.recency import RecencyAnalysis
from interestingness_xrl.explainability.analysis.meta.transition_values import TransitionValuesAnalysis
from interestingness_xrl.explainability.analysis.meta.sequences import SequenceAnalysis
from interestingness_xrl.explainability.analysis.meta.contradictions import ContradictionAnalysis
from interestingness_xrl.util import print_line


class FullAnalysis(AnalysisBase):
    """
    Represents a complete or full analysis, i.e., containing all possible analyses that can be performed.
    """

    def __init__(self, helper, agent, trans_analysis, rwd_analysis, state_freq_analysis, state_action_freq_analysis,
                 value_analysis, recency_analysis, trans_value_analysis, sequence_analysis, contradiction_analysis):
        """
        :param ScenarioHelper helper: the environment helper containing all necessary config and methods.
        :param QValueBasedAgent agent: the agent to be analyzed.
        :param TransitionAnalysis trans_analysis: the analysis over transitions.
        :param RewardAnalysis rwd_analysis: the analysis over rewards.
        :param StateFrequencyAnalysis state_freq_analysis: the analysis over state frequencies.
        :param StateActionFrequencyAnalysis state_action_freq_analysis: the analysis over state-action frequencies.
        :param ValueAnalysis value_analysis: the analysis over values.
        :param RecencyAnalysis recency_analysis: the analysis over state recency.
        :param TransitionValuesAnalysis trans_value_analysis: the analysis over values of transitions.
        :param SequenceAnalysis sequence_analysis: the analysis over sequences.
        :param ContradictionAnalysis contradiction_analysis: the analysis over contradictions.
        """
        super().__init__(helper, agent)
        self.trans_analysis = trans_analysis
        self.rwd_analysis = rwd_analysis
        self.state_freq_analysis = state_freq_analysis
        self.state_action_freq_analysis = state_action_freq_analysis
        self.value_analysis = value_analysis
        self.recency_analysis = recency_analysis
        self.trans_value_analysis = trans_value_analysis
        self.sequence_analysis = sequence_analysis
        self.contradiction_analysis = contradiction_analysis

    def analyze(self):
        pass

    def difference_to(self, other):
        """
        Gets the difference between this analysis and another given analysis, i.e., the elements that are new and
        therefore missing in the other analysis. This can be used to highlight what is new in this analysis compared to
        one that was performed at a prior stage of the agent's behavior / learning.
        :param FullAnalysis other: the other analysis to get the difference to.
        :return FullAnalysis: an analysis resulting from the difference between this and the given analysis.
        """
        trans_analysis = self.trans_analysis.difference_to(other.trans_analysis)
        rwd_analysis = self.rwd_analysis.difference_to(other.rwd_analysis)
        state_freq_analysis = self.state_freq_analysis.difference_to(other.state_freq_analysis)
        state_action_freq_analysis = self.state_action_freq_analysis.difference_to(other.state_action_freq_analysis)
        value_analysis = self.value_analysis.difference_to(other.value_analysis)
        recency_analysis = self.recency_analysis.difference_to(other.recency_analysis)
        trans_value_analysis = self.trans_value_analysis.difference_to(other.trans_value_analysis)
        sequence_analysis = self.sequence_analysis.difference_to(other.sequence_analysis)
        contradiction_analysis = self.contradiction_analysis.difference_to(other.contradiction_analysis)

        return FullAnalysis(self.helper, self.agent,
                            trans_analysis, rwd_analysis, state_freq_analysis, state_action_freq_analysis,
                            value_analysis, recency_analysis, trans_value_analysis, sequence_analysis,
                            contradiction_analysis)

    def set_helper(self, helper):
        """
        Sets all analyses configuration.
        :param ScenarioHelper helper: the environment helper containing all necessary config and methods.
        :return:
        """
        self.helper = \
            self.trans_analysis.helper = \
            self.rwd_analysis.helper = \
            self.state_freq_analysis.helper = \
            self.state_action_freq_analysis.helper = \
            self.value_analysis.helper = \
            self.recency_analysis.helper = \
            self.trans_value_analysis.helper = \
            self.sequence_analysis.helper = \
            self.contradiction_analysis.helper = helper

        self.config = \
            self.trans_analysis.config = \
            self.rwd_analysis.config = \
            self.state_freq_analysis.config = \
            self.state_action_freq_analysis.config = \
            self.value_analysis.config = \
            self.recency_analysis.config = \
            self.trans_value_analysis.config = \
            self.sequence_analysis.config = \
            self.contradiction_analysis.config = helper.config

    def _save_report(self, file, write_console):
        print_line('\n===================================================================', file, write_console)
        print_line('TRANSITION ANALYSIS', file, write_console)
        self.trans_analysis._save_report(file, write_console)

        print_line('\n===================================================================', file, write_console)
        print_line('REWARD ANALYSIS', file, write_console)
        self.rwd_analysis._save_report(file, write_console)

        print_line('\n===================================================================', file, write_console)
        print_line('STATE FREQUENCY ANALYSIS', file, write_console)
        self.state_freq_analysis._save_report(file, write_console)

        print_line('\n===================================================================', file, write_console)
        print_line('STATE-ACTION FREQUENCY ANALYSIS', file, write_console)
        self.state_action_freq_analysis._save_report(file, write_console)

        print_line('\n===================================================================', file, write_console)
        print_line('VALUE ANALYSIS', file, write_console)
        self.value_analysis._save_report(file, write_console)

        print_line('\n===================================================================', file, write_console)
        print_line('RECENCY ANALYSIS', file, write_console)
        self.recency_analysis._save_report(file, write_console)

        print_line('\n===================================================================', file, write_console)
        print_line('TRANSITION VALUE ANALYSIS', file, write_console)
        self.trans_value_analysis._save_report(file, write_console)

        print_line('\n===================================================================', file, write_console)
        print_line('SEQUENCE ANALYSIS', file, write_console)
        self.sequence_analysis._save_report(file, write_console)

        print_line('\n===================================================================', file, write_console)
        print_line('CONTRADICTION ANALYSIS', file, write_console)
        self.contradiction_analysis._save_report(file, write_console)

    def _save_visual_report(self, path):
        self.trans_analysis._save_visual_report(path)
        self.rwd_analysis._save_visual_report(path)
        self.state_freq_analysis._save_visual_report(path)
        self.state_action_freq_analysis._save_visual_report(path)
        self.value_analysis._save_visual_report(path)
        self.recency_analysis._save_visual_report(path)
        self.trans_value_analysis._save_visual_report(path)
        self.sequence_analysis._save_visual_report(path)
        self.contradiction_analysis._save_visual_report(path)

    def get_sample_interesting_aspects(self, s, a, r, ns):
        aspects = []
        aspects.extend(self.trans_analysis.get_sample_interesting_aspects(s, a, r, ns))
        aspects.extend(self.rwd_analysis.get_sample_interesting_aspects(s, a, r, ns))
        aspects.extend(self.state_freq_analysis.get_sample_interesting_aspects(s, a, r, ns))
        aspects.extend(self.state_action_freq_analysis.get_sample_interesting_aspects(s, a, r, ns))
        aspects.extend(self.value_analysis.get_sample_interesting_aspects(s, a, r, ns))
        aspects.extend(self.trans_value_analysis.get_sample_interesting_aspects(s, a, r, ns))
        aspects.extend(self.recency_analysis.get_sample_interesting_aspects(s, a, r, ns))
        aspects.extend(self.sequence_analysis.get_sample_interesting_aspects(s, a, r, ns))
        aspects.extend(self.contradiction_analysis.get_sample_interesting_aspects(s, a, r, ns))
        return aspects

    def get_interesting_aspects_grouped(self, s, a, r, ns):
        """
        Gets all interesting aspects found in the given full analysis regarding the given sample, grouped by analysis type.
        :param int s: the sample's initial state.
        :param int a: the sample's action.
        :param float r: the sample's reward.
        :param int ns: the sample's next state.
        :rtype: dict
        :return: a dictionary containing all interesting aspects found in the analyses regarding the given sample.
        """
        return {
            '0-transitions': self.trans_analysis.get_sample_interesting_aspects(s, a, r, ns),
            '0-rewards': self.rwd_analysis.get_sample_interesting_aspects(s, a, r, ns),
            '1-state-frequency': self.state_freq_analysis.get_sample_interesting_aspects(s, a, r, ns),
            '1-action-frequency': self.state_action_freq_analysis.get_sample_interesting_aspects(s, a, r, ns),
            '1-values': self.value_analysis.get_sample_interesting_aspects(s, a, r, ns),
            '1-recency': self.recency_analysis.get_sample_interesting_aspects(s, a, r, ns),
            '2-transition-values': self.trans_value_analysis.get_sample_interesting_aspects(s, a, r, ns),
            '2-sequences': self.sequence_analysis.get_sample_interesting_aspects(s, a, r, ns),
            '2-contradictions': self.contradiction_analysis.get_sample_interesting_aspects(s, a, r, ns)
        }

    def get_interestingness_names(self):
        names = []
        names.extend(self.trans_analysis.get_interestingness_names())
        names.extend(self.rwd_analysis.get_interestingness_names())
        names.extend(self.state_freq_analysis.get_interestingness_names())
        names.extend(self.state_action_freq_analysis.get_interestingness_names())
        names.extend(self.value_analysis.get_interestingness_names())
        names.extend(self.trans_value_analysis.get_interestingness_names())
        names.extend(self.recency_analysis.get_interestingness_names())
        names.extend(self.sequence_analysis.get_interestingness_names())
        names.extend(self.contradiction_analysis.get_interestingness_names())
        return names

    def get_interestingness_names_grouped(self):
        """
        Gets all interesting aspects possibly found by all other analyses, grouped by analysis type.
        :rtype: dict
        :return: a dictionary containing all interesting aspects possibly found in the analyses.
        """
        return {
            '0-transitions': self.trans_analysis.get_interestingness_names(),
            '0-rewards': self.rwd_analysis.get_interestingness_names(),
            '1-state-frequency': self.state_freq_analysis.get_interestingness_names(),
            '1-action-frequency': self.state_action_freq_analysis.get_interestingness_names(),
            '1-values': self.value_analysis.get_interestingness_names(),
            '1-recency': self.recency_analysis.get_interestingness_names(),
            '2-transition-values': self.trans_value_analysis.get_interestingness_names(),
            '2-sequences': self.sequence_analysis.get_interestingness_names(),
            '2-contradictions': self.contradiction_analysis.get_interestingness_names()
        }

    def get_stats(self):
        stats = {}
        stats.update(self.trans_analysis.get_stats())
        stats.update(self.rwd_analysis.get_stats())
        stats.update(self.state_freq_analysis.get_stats())
        stats.update(self.state_action_freq_analysis.get_stats())
        stats.update(self.value_analysis.get_stats())
        stats.update(self.trans_value_analysis.get_stats())
        stats.update(self.recency_analysis.get_stats())
        stats.update(self.sequence_analysis.get_stats())
        stats.update(self.contradiction_analysis.get_stats())
        return stats

    def get_stats_grouped(self):
        """
        Gets the analysis stats grouped by analysis type.
        :rtype: dict
        :return: a dictionary containing all stats found in the analyses grouped by analysis type.
        """
        return {
            '0-transitions': self.trans_analysis.get_stats(),
            '0-rewards': self.rwd_analysis.get_stats(),
            '1-state-frequency': self.state_freq_analysis.get_stats(),
            '1-action-frequency': self.state_action_freq_analysis.get_stats(),
            '1-values': self.value_analysis.get_stats(),
            '1-recency': self.recency_analysis.get_stats(),
            '2-transition-values': self.trans_value_analysis.get_stats(),
            '2-sequences': self.sequence_analysis.get_stats(),
            '2-contradictions': self.contradiction_analysis.get_stats()
        }

    def save_json(self, json_file_path):
        # "hides" the agent and config

        self.trans_analysis.helper = \
            self.rwd_analysis.helper = \
            self.state_freq_analysis.helper = \
            self.state_action_freq_analysis.helper = \
            self.value_analysis.helper = \
            self.recency_analysis.helper = \
            self.trans_value_analysis.helper = \
            self.sequence_analysis.helper = \
            self.contradiction_analysis.helper = None

        self.trans_analysis.config = \
            self.rwd_analysis.config = \
            self.state_freq_analysis.config = \
            self.state_action_freq_analysis.config = \
            self.value_analysis.config = \
            self.recency_analysis.config = \
            self.trans_value_analysis.config = \
            self.sequence_analysis.config = \
            self.contradiction_analysis.config = None

        self.trans_analysis.agent = \
            self.rwd_analysis.agent = \
            self.state_freq_analysis.agent = \
            self.state_action_freq_analysis.agent = \
            self.value_analysis.agent = \
            self.recency_analysis.agent = \
            self.trans_value_analysis.agent = \
            self.sequence_analysis.agent = \
            self.contradiction_analysis.agent = None

        # saves all elements to json
        super().save_json(json_file_path)

        # puts agents and configs back
        self.trans_analysis.helper = \
            self.rwd_analysis.helper = \
            self.state_freq_analysis.helper = \
            self.state_action_freq_analysis.helper = \
            self.value_analysis.helper = \
            self.recency_analysis.helper = \
            self.trans_value_analysis.helper = \
            self.sequence_analysis.helper = \
            self.contradiction_analysis.helper = self.helper

        self.trans_analysis.config = \
            self.rwd_analysis.config = \
            self.state_freq_analysis.config = \
            self.state_action_freq_analysis.config = \
            self.value_analysis.config = \
            self.recency_analysis.config = \
            self.trans_value_analysis.config = \
            self.sequence_analysis.config = \
            self.contradiction_analysis.config = self.config

        self.trans_analysis.agent = \
            self.rwd_analysis.agent = \
            self.state_freq_analysis.agent = \
            self.state_action_freq_analysis.agent = \
            self.value_analysis.agent = \
            self.recency_analysis.agent = \
            self.trans_value_analysis.agent = \
            self.sequence_analysis.agent = \
            self.contradiction_analysis.agent = self.agent

    def _after_loaded_json(self):
        self.trans_analysis._after_loaded_json()
        self.rwd_analysis._after_loaded_json()
        self.state_freq_analysis._after_loaded_json()
        self.state_action_freq_analysis._after_loaded_json()
        self.value_analysis._after_loaded_json()
        self.recency_analysis._after_loaded_json()
        self.trans_value_analysis._after_loaded_json()
        self.sequence_analysis._after_loaded_json()
        self.contradiction_analysis._after_loaded_json()
