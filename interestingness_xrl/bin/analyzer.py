__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import argparse
from os import makedirs
from os.path import join, exists
from shutil import rmtree
from interestingness_xrl.learning.agents import QLearningAgent
from interestingness_xrl.scenarios import AgentType, get_agent_output_dir, DEFAULT_CONFIG, create_helper, \
    get_analysis_config, get_analysis_output_dir
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration, AnalysisConfiguration
from interestingness_xrl.explainability.analysis import AnalysisBase
from interestingness_xrl.explainability.analysis.full_analysis import FullAnalysis
from interestingness_xrl.explainability.analysis.task.transitions import TransitionAnalysis
from interestingness_xrl.explainability.analysis.task.rewards import RewardAnalysis
from interestingness_xrl.explainability.analysis.interaction.state_frequency import StateFrequencyAnalysis
from interestingness_xrl.explainability.analysis.interaction.action_frequency import StateActionFrequencyAnalysis
from interestingness_xrl.explainability.analysis.interaction.values import ValueAnalysis
from interestingness_xrl.explainability.analysis.interaction.recency import RecencyAnalysis
from interestingness_xrl.explainability.analysis.meta.transition_values import TransitionValuesAnalysis
from interestingness_xrl.explainability.analysis.meta.sequences import SequenceAnalysis
from interestingness_xrl.explainability.analysis.meta.contradictions import ContradictionAnalysis

DEF_AGENT_TYPE = AgentType.Testing
CLEAR_RESULTS = True


def analyze(analysis, output_dir, name):
    """
    Performs the given analysis, saves results to json, text and visual reports.
    :param AnalysisBase analysis: the analysis to be performed.
    :param str output_dir: the output directory in which to save results.
    :param str name: the name of the analysis used for file and directory naming.
    :return:
    """
    analysis.analyze()
    results_dir = join(output_dir, name)
    if not exists(results_dir):
        makedirs(results_dir)

    analysis.save_visual_report(results_dir, False)
    analysis.save_json(join(results_dir, '{}.json'.format(name)))
    analysis.save_report(join(results_dir, '{}.txt'.format(name)), True)

    print('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='RL interestingness analyzer')
    parser.add_argument('-a', '--agent', help='agent type', type=int, default=DEF_AGENT_TYPE)
    parser.add_argument('-o', '--output', help='directory in which to store results')
    parser.add_argument('-c', '--config', help='path to config file')
    args = parser.parse_args()

    # loads environment config from results dir
    agent_dir = get_agent_output_dir(DEFAULT_CONFIG, args.agent)
    if not exists(agent_dir):
        raise ValueError('Could not load agent from: {}'.format(agent_dir))

    config_file = join(agent_dir, 'config.json')
    if not exists(config_file):
        raise ValueError('Configuration not found: {}'.format(config_file))
    env_config = EnvironmentConfiguration.load_json(config_file)

    # creates env helper
    helper = create_helper(env_config)

    # tries to load analysis config from given file
    an_config = get_analysis_config(env_config)
    if args.config is not None and exists(args.config):
        an_config = AnalysisConfiguration.load_json(args.config)

    # creates an agent and loads all tables
    agent = QLearningAgent(env_config.num_states, env_config.num_actions)
    agent.load(agent_dir)

    # creates output dir if needed
    output_dir = args.output if args.output is not None else get_analysis_output_dir(agent_dir)
    if not exists(output_dir):
        makedirs(output_dir)
    elif CLEAR_RESULTS:
        rmtree(output_dir)
        makedirs(output_dir)

    # saves / copies configs to file
    env_config.save_json(join(output_dir, 'config.json'))
    an_config.save_json(join(output_dir, 'analysis_config.json'))

    # ===================================================================
    # LEVEL 0

    trans_analysis = TransitionAnalysis(helper, agent, an_config.min_count, an_config.trans_min_states,
                                        an_config.certain_trans_max_disp, an_config.uncertain_trans_min_disp,
                                        an_config.certain_trans_max_disp, an_config.uncertain_trans_min_disp,
                                        an_config.certain_trans_max_disp, an_config.uncertain_trans_min_disp)
    analyze(trans_analysis, output_dir, '0-transitions')

    rwd_analysis = RewardAnalysis(
        helper, agent,
        an_config.min_count, an_config.rwd_outlier_stds, an_config.rwd_outlier_stds)
    analyze(rwd_analysis, output_dir, '0-rewards')

    # ===================================================================
    # LEVEL 1

    state_freq_analysis = StateFrequencyAnalysis(
        helper, agent, an_config.freq_min_state_count, an_config.infreq_max_state_count,
        an_config.min_feat_set_count, an_config.assoc_min_feat_set_jacc,
        an_config.assoc_min_feat_rule_conf, an_config.no_assoc_max_feat_set_jacc)
    analyze(state_freq_analysis, output_dir, '1-state-frequency')

    state_action_freq_analysis = StateActionFrequencyAnalysis(
        helper, agent, an_config.min_count,
        an_config.certain_exec_max_disp, an_config.uncertain_exec_min_disp,
        an_config.certain_exec_max_disp, an_config.uncertain_exec_min_disp)
    analyze(state_action_freq_analysis, output_dir, '1-action-frequency')

    value_analysis = ValueAnalysis(
        helper, agent, an_config.min_count,
        an_config.value_outlier_stds, an_config.pred_error_outlier_stds, an_config.value_outlier_stds)
    analyze(value_analysis, output_dir, '1-values')

    recency_analysis = RecencyAnalysis(
        helper, agent, an_config.min_count, an_config.max_time_step, an_config.max_time_step)
    analyze(recency_analysis, output_dir, '1-recency')

    # ===================================================================
    # LEVEL 2

    trans_value_analysis = TransitionValuesAnalysis(
        helper, agent, an_config.min_count, an_config.min_count,
        an_config.val_diff_var_outlier_stds)
    analyze(trans_value_analysis, output_dir, '2-transition-values')

    sequence_analysis = SequenceAnalysis(
        helper, agent, state_freq_analysis, state_action_freq_analysis, trans_analysis, trans_value_analysis,
        recency_analysis)
    analyze(sequence_analysis, output_dir, '2-sequences')

    contradiction_analysis = ContradictionAnalysis(
        helper, agent, trans_value_analysis, state_action_freq_analysis,
        an_config.min_count, an_config.action_jsd_threshold)
    analyze(contradiction_analysis, output_dir, '2-contradictions')

    # ===================================================================
    # putting everything together

    full_analysis = FullAnalysis(
        helper, agent, trans_analysis, rwd_analysis, state_freq_analysis, state_action_freq_analysis,
        value_analysis, recency_analysis, trans_value_analysis, sequence_analysis, contradiction_analysis)
    full_analysis.save_json(join(output_dir, 'full-analysis.json'))
    # full_analysis.save_report(join(output_dir, 'full-analysis.txt'), True)
    # full_analysis.save_visual_report(join(output_dir, 'full-analysis'))
