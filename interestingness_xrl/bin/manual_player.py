__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import sys
import gym
import pygame
from os import makedirs
from os.path import exists, join
from gym.wrappers import Monitor
from frogger import ACTION_NO_MOVE_KEY, INVALID_ACTION_KEY
from interestingness_xrl.util import clean_console, save_image
from interestingness_xrl.learning import get_features_from_index
from interestingness_xrl.learning.behavior_tracker import BehaviorTracker
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration
from interestingness_xrl.scenarios import DEFAULT_CONFIG, create_helper, get_agent_output_dir, AgentType, create_agent

REAL_TIME = False
FPS = 15
SHOW_SCORE_BAR = True
SOUND = False


def process_keys():
    global save_features, save_environment

    # prevent events from going to the game
    pressed = []
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
                save_features = True
            elif event.key == pygame.K_e:
                save_environment = True
            else:
                pressed.append(event.key)
    return pressed


def select_action(actions_keys):
    # checks if pressed key corresponds to any action
    pressed = process_keys()
    if REAL_TIME:
        pressed.append(ACTION_NO_MOVE_KEY)
    for _a, action in enumerate(actions_keys):
        if action in pressed:
            return _a
    return -1


if __name__ == '__main__':

    # tries to get agent type
    agent_t = int(sys.argv[1]) if len(sys.argv) > 1 else AgentType.Manual

    # tries to load agent from results dir
    agent_dir = get_agent_output_dir(DEFAULT_CONFIG, agent_t)
    config_file = join(agent_dir, 'config.json')
    if agent_t == AgentType.Manual:
        # tries to load env config from given file
        config_file = sys.argv[1] if len(sys.argv) > 1 else None
        config = DEFAULT_CONFIG if config_file is None or not exists(config_file) \
            else EnvironmentConfiguration.load_json(config_file)
    elif not exists(agent_dir):
        raise ValueError('Could not load agent from: {}'.format(agent_dir))
    elif not exists(config_file):
        raise ValueError('Configuration not found: {}'.format(config_file))
    else:
        config = EnvironmentConfiguration.load_json(config_file)

    # adds no-op
    if REAL_TIME and config.num_actions == 4:
        config.actions['no-op'] = ACTION_NO_MOVE_KEY
        config.num_actions += 1
    actions = list(config.actions.values())
    action_names = config.get_action_names()

    # creates env helper
    helper = create_helper(config, SOUND)
    feats_nbins = helper.get_features_bins()

    # checks for provided output dir
    output_dir = sys.argv[2] if len(sys.argv) > 2 else get_agent_output_dir(config, AgentType.Manual)
    if not exists(output_dir):
        makedirs(output_dir)

    # register environment in Gym according to env_config
    env_id = '{}-manual-v0'.format(config.gym_env_id)
    helper.register_gym_environment(env_id, True, FPS, SHOW_SCORE_BAR)

    # saves / copies configs to file
    config.save_json(join(output_dir, 'config.json'))
    helper.save_state_features(join(output_dir, 'state_features.csv'))

    # create environment and monitor
    env = gym.make(env_id)
    env = Monitor(env, directory=output_dir, force=True, video_callable=lambda _: True)
    env.seed(config.seed)

    # adds reference to monitor to allow for gym environments to update video frames
    env.env.env.monitor = env

    # create the agent
    agent, exploration_strategy = create_agent(helper, AgentType.Manual, None)
    behavior_tracker = BehaviorTracker(config.num_episodes)

    # tries to load agent info
    if agent_t != AgentType.Manual and exists(agent_dir):
        agent.load(agent_dir)

    window_still_open = True
    e = 0
    save_features = save_environment = False
    while window_still_open and e < config.num_episodes:

        # reset environment
        old_obs = env.reset()
        old_s = helper.get_state_from_observation(old_obs, 0, False)

        helper.update_stats_episode(e)
        exploration_strategy.update(e)

        t = 0
        done = False
        while not done:

            # select action
            a = INVALID_ACTION_KEY
            while a == INVALID_ACTION_KEY:
                a = select_action(actions)
                pygame.event.pump()

            exploration_strategy.set_action(a)

            # observe transition and reward
            obs, r, done, _ = env.step(a)
            s = helper.get_state_from_observation(obs, r, done)
            r = helper.get_reward(old_s, a, r, s, done)

            # learn and update stats
            if agent_t == AgentType.Manual and a != INVALID_ACTION_KEY and a != ACTION_NO_MOVE_KEY:
                agent.update(old_s, a, r, s)
            behavior_tracker.add_sample(old_s, a)
            helper.update_stats(e, t, old_obs, obs, old_s, a, r, s)

            obs_vec = get_features_from_index(s, feats_nbins)

            # prints information
            clean_console()
            print('Action: {}, reward: {}'.format(action_names[a], r))
            print('------------------------------------------------------------')
            helper.print_features(obs_vec, True)
            print('------------------------------------------------------------')
            for act_name in action_names:
                print(act_name.upper().ljust(15), end='\t')
            print()
            for i, act_name in enumerate(action_names):
                print('{:10.5f}'.format(agent.q[s][i]), end='\t')
            print()

            # checks if save feature was pressed
            if save_features:
                helper.save_features_image(obs_vec, join(output_dir, 'features.png'))
                save_features = False

            # checks if save environment was pressed
            if save_environment:
                save_image(env, join(output_dir, 'environment.png'))
                save_environment = False

            window_still_open = env.render() is not None
            old_s = s
            old_obs = obs
            t += 1

        # signals new episode to tracker
        behavior_tracker.new_episode()
        e += 1

    # writes results to files
    behavior_tracker.save(output_dir)
    helper.save_stats(join(output_dir, 'results'))

    print('\nResults written to:\n\t\'{}\''.format(output_dir))
    env.close()
