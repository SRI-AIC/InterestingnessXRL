__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import argparse
import sys
import gym
from os import makedirs
from os.path import join, exists
from shutil import rmtree
from interestingness_xrl.util import clean_console
from interestingness_xrl.learning import read_table_csv
from interestingness_xrl.explainability.analysis.full_analysis import FullAnalysis
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration
from interestingness_xrl.learning.behavior_tracker import BehaviorTracker
from interestingness_xrl.scenarios import DEFAULT_CONFIG, get_agent_output_dir, create_helper, AgentType, \
    ReportType, create_agent, get_analysis_output_dir, get_explanation_output_dir, create_explainer

# DEF_AGENT_TYPE = AgentType.Learning
DEF_AGENT_TYPE = AgentType.Testing
DEF_REPORT_TYPE = ReportType.Highlights

CLEAN_DIR = True


class ExplainerMonitor(object):
    def __init__(self):
        # just sets a property with this name so that the explainer can capture environment image frames
        self.video_recorder = explainer


def run_episodes(name, new_episode_func, update_func):
    global behavior_tracker, helper

    for e in range(behavior_tracker.num_episodes):

        # gets state and action sequence
        s_seq = behavior_tracker.s_s[e]
        a_seq = behavior_tracker.s_a[e]
        seq_len = len(s_seq)

        # signals new episode
        new_episode_func(e, seq_len)

        old_obs = env.reset()
        helper.get_state_from_observation(old_obs, 0, False)

        clean_console()
        print('Processing {} episode {}...'.format(name, e))

        for t in range(seq_len):

            # gets state-action pair
            old_s = s_seq[t]
            a = a_seq[t]

            # steps the environment, gets next state
            obs, r, done, _ = env.step(a)
            s = helper.get_state_from_observation(obs, r, done)
            r = helper.get_reward(old_s, a, r, s, done)

            # checks possible synchronization errors
            if done and t != seq_len - 1:
                raise ValueError(
                    'Environment ended at {}, before tracked behavior which ended at: {}'.format(t, seq_len - 1))
            if t == seq_len - 1 and not done:
                raise ValueError('Environment did not end at {} like it was supposed to'.format(t))
            if not done and s != s_seq[t + 1]:
                raise ValueError('Environment state {} does not match tracked state {}'.format(s, s_seq[t + 1]))

            if name == 'analysis' and explanation_t == ReportType.Heatmaps:
                helper.update_stats(e, t, old_obs, obs, old_s, a, r, s)
            update_func(t, old_obs, old_s, a, r, s)

            old_obs = obs


if __name__ == '__main__':

    # tries to get explanation and agent types
    parser = argparse.ArgumentParser(description='Elements Reporting')
    parser.add_argument('-a', '--agent', help='agent type', type=int, default=DEF_AGENT_TYPE)
    parser.add_argument('-r', '--report', help='report type', type=int, default=DEF_REPORT_TYPE)
    args = parser.parse_args()

    agent_t = args.agent
    explanation_t = args.report

    # tries to load agent from results dir
    agent_dir = get_agent_output_dir(DEFAULT_CONFIG, agent_t)
    if not exists(agent_dir):
        raise ValueError('Could not load agent from: {}'.format(agent_dir))

    config_file = join(agent_dir, 'config.json')
    if not exists(config_file):
        raise ValueError('Configuration not found: {}'.format(config_file))
    config = EnvironmentConfiguration.load_json(config_file)

    # creates env helper
    helper = create_helper(config)

    # tries to load full analysis
    analyses_dir = get_analysis_output_dir(agent_dir)
    file_name = join(analyses_dir, 'full-analysis.json')
    if exists(file_name):
        full_analysis = FullAnalysis.load_json(file_name)
    else:
        raise ValueError('Full analysis not found: {}'.format(file_name))
    full_analysis.set_helper(helper)

    # creates and load the agent's tables
    agent, exploration_strategy = create_agent(helper, DEF_AGENT_TYPE, None)
    agent.load(agent_dir)

    # loads the agent's behavior
    behavior_tracker = BehaviorTracker(0)
    behavior_tracker.load(agent_dir)

    # tries to load recorded episodes from results dir
    rec_episodes_file = join(agent_dir, 'rec_episodes.csv')
    if not exists(rec_episodes_file):
        raise ValueError('Recorded episodes file not found: {}'.format(rec_episodes_file))
    recorded_episodes = read_table_csv(rec_episodes_file, dtype=int)

    # creates output dir if needed
    output_dir = get_explanation_output_dir(agent_dir, explanation_t)
    if not exists(output_dir):
        makedirs(output_dir)
    elif CLEAN_DIR:
        rmtree(output_dir)
        makedirs(output_dir)

    # saves / copies configs to file
    config.save_json(join(output_dir, 'config.json'))

    # register environment in Gym according to env config
    env_id = '{}-report-v0'.format(config.gym_env_id)
    helper.register_gym_environment(env_id, False, 0, False)

    # create environment
    env = gym.make(env_id)

    # initializes seed
    env.seed(config.seed)

    # creates and initializes explainer
    explainer = create_explainer(explanation_t, env, helper, full_analysis, output_dir, recorded_episodes)

    # adds reference to monitor to allow for gym environments to update video frames
    monitor = ExplainerMonitor()
    env.env.monitor = explainer.monitor = monitor

    # runs analysis episodes
    run_episodes('analysis', explainer.new_analysis_episode, explainer.update_analysis)

    explainer.finalize_analysis()

    # re-initializes env
    env.seed(config.seed)

    # runs explanation episodes
    run_episodes('explanation', explainer.new_explain_episode, explainer.update_explanation)

    # closes the explainer, saving results as needed
    explainer.close()
