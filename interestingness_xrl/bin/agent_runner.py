__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import gym
import numpy as np
import argparse
from os import makedirs
from os.path import join, exists
from gym.wrappers import Monitor
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration
from interestingness_xrl.learning import write_table_csv
from interestingness_xrl.learning.behavior_tracker import BehaviorTracker
from interestingness_xrl.scenarios import DEFAULT_CONFIG, create_helper, AgentType, get_agent_output_dir, create_agent

DEF_AGENT_TYPE = AgentType.Learning
DEF_TRIAL = 0

FPS = 20
SHOW_SCORE_BAR = True
CLEAR_RESULTS = True


def video_schedule(config, videos):
    # linear capture schedule
    return lambda e: videos and \
                     (e == config.num_episodes - 1 or e % int(config.num_episodes / config.num_recorded_videos) == 0)


def run_trial(args):
    # tries to get agent type
    agent_t = args.agent
    results_dir = ''
    if agent_t == AgentType.Testing:

        # tries to load config from provided results dir path
        results_dir = args.results if args.results is not None else \
            get_agent_output_dir(DEFAULT_CONFIG, AgentType.Learning)
        config_file = join(results_dir, 'config.json')
        if not exists(results_dir) or not exists(config_file):
            raise ValueError('Could not load configuration from: {}.'.format(config_file))
        config = EnvironmentConfiguration.load_json(config_file)

        # if testing, we want to force a seed different than training (diff. test environments)
        config.seed += 1

    else:
        # tries to load env config from provided file path
        config_file = args.config
        config = DEFAULT_CONFIG if config_file is None or not exists(config_file) \
            else EnvironmentConfiguration.load_json(config_file)

    # creates env helper
    helper = create_helper(config)

    # checks for provided output dir
    output_dir = args.output if args.output is not None else get_agent_output_dir(config, agent_t, args.trial)
    if not exists(output_dir):
        makedirs(output_dir)

    # saves / copies configs to file
    config.save_json(join(output_dir, 'config.json'))
    helper.save_state_features(join(output_dir, 'state_features.csv'))

    # register environment in Gym according to env config
    env_id = '{}-{}-v0'.format(config.gym_env_id, args.trial)
    helper.register_gym_environment(env_id, False, FPS, SHOW_SCORE_BAR)

    # create environment and monitor
    env = gym.make(env_id)
    # todo
    config.num_episodes = 100
    video_callable = video_schedule(config, args.record)
    env = Monitor(env, directory=output_dir, force=True, video_callable=video_callable)

    # adds reference to monitor to allow for gym environments to update video frames
    if video_callable(0):
        env.env.monitor = env

    # initialize seeds (one for the environment, another for the agent)
    env.seed(config.seed + args.trial)
    agent_rng = np.random.RandomState(config.seed + args.trial)

    # creates the agent
    agent, exploration_strategy = create_agent(helper, agent_t, agent_rng)

    # if testing, loads tables from file (some will be filled by the agent during the interaction)
    if agent_t == AgentType.Testing:
        agent.load(results_dir, )

    # runs episodes
    behavior_tracker = BehaviorTracker(config.num_episodes)
    recorded_episodes = []
    for e in range(config.num_episodes):

        # checks whether to activate video monitoring
        env.env.monitor = env if video_callable(e) else None

        # reset environment
        old_obs = env.reset()
        old_s = helper.get_state_from_observation(old_obs, 0, False)

        if args.verbose:
            helper.update_stats_episode(e)
        exploration_strategy.update(e)

        t = 0
        done = False
        while not done:
            # select action
            a = agent.act(old_s)

            # observe transition
            obs, r, done, _ = env.step(a)
            s = helper.get_state_from_observation(obs, r, done)
            r = helper.get_reward(old_s, a, r, s, done)

            # update agent and stats
            agent.update(old_s, a, r, s)
            behavior_tracker.add_sample(old_s, a)
            helper.update_stats(e, t, old_obs, obs, old_s, a, r, s)

            old_s = s
            old_obs = obs
            t += 1

        # adds to recorded episodes list
        if video_callable(e):
            recorded_episodes.append(e)

        # signals new episode to tracker
        behavior_tracker.new_episode()

    # writes results to files
    agent.save(output_dir)
    behavior_tracker.save(output_dir)
    write_table_csv(recorded_episodes, join(output_dir, 'rec_episodes.csv'))
    helper.save_stats(join(output_dir, 'results'), CLEAR_RESULTS)
    print('\nResults of trial {} written to:\n\t\'{}\''.format(args.trial, output_dir))

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent runner')
    parser.add_argument('-a', '--agent', help='agent type', type=int, default=DEF_AGENT_TYPE)
    parser.add_argument('-o', '--output', help='directory in which to store results')
    parser.add_argument('-r', '--results', help='directory from which to load results')
    parser.add_argument('-c', '--config', help='path to config file')
    parser.add_argument('-t', '--trial', help='trial number to run', type=int, default=DEF_TRIAL)
    parser.add_argument('-rv', '--record', help='record videos according to linear schedule', action='store_true')
    parser.add_argument('-v', '--verbose', help='print information to the console', action='store_true')
    args = parser.parse_args()

    # runs trial
    run_trial(args)
