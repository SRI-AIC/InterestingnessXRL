__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import sys
import gym
import pygame
from os.path import join, exists
from interestingness_xrl.util import clean_console
from interestingness_xrl.learning import get_features_from_index
from interestingness_xrl.learning.agents import QValueBasedAgent
from interestingness_xrl.learning.behavior_tracker import BehaviorTracker
from interestingness_xrl.scenarios import DEFAULT_CONFIG, get_agent_output_dir, AgentType, create_helper
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration

# AGENT_TYPE = AgentType.Learning
DEF_AGENT_TYPE = AgentType.Testing
# START_EPISODE = 1999
START_EPISODE = 0

FPS = 2000
SHOW_SCORE_BAR = True
SOUND = False


def process_keys():
    global advance_time, advance_episode

    # prevent events from going to the game
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                advance_time = True
            elif event.key == pygame.K_s:
                advance_episode = True


if __name__ == '__main__':

    # tries to get agent type
    agent_t = int(sys.argv[1]) if len(sys.argv) > 1 else DEF_AGENT_TYPE

    # tries to load agent from results dir
    agent_dir = sys.argv[2] if len(sys.argv) > 2 else get_agent_output_dir(DEFAULT_CONFIG, agent_t)
    if not exists(agent_dir):
        raise ValueError('Could not load agent from: {}'.format(agent_dir))

    config_file = join(agent_dir, 'config.json')
    if not exists(config_file):
        raise ValueError('Configuration not found: {}'.format(config_file))
    config = EnvironmentConfiguration.load_json(config_file)
    action_names = config.get_action_names()

    # creates env helper
    helper = create_helper(config, SOUND)
    feats_nbins = helper.get_features_bins()

    # loads the agent's behavior
    behavior_tracker = BehaviorTracker(0)
    behavior_tracker.load(agent_dir)

    # register environment in Gym according to env_config
    env_id = '{}-replay-v0'.format(config.gym_env_id)
    helper.register_gym_environment(env_id, True, FPS, SHOW_SCORE_BAR)

    # creates environment
    env = gym.make(env_id)

    # initializes seed according to config
    env.seed(config.seed)

    # creates and loads agent
    agent = QValueBasedAgent(config.num_states, config.num_actions, action_names=action_names)
    agent.load(agent_dir)

    # runs all episodes
    advance_time = advance_episode = False
    for e in range(behavior_tracker.num_episodes):

        # gets state and action sequence
        s_seq = behavior_tracker.s_s[e]
        a_seq = behavior_tracker.s_a[e]

        advance_episode = e < START_EPISODE
        old_obs = env.reset()
        old_s = helper.get_state_from_observation(old_obs, 0, False)

        seq_len = len(s_seq)
        clean_console()
        print('Replaying episode {} ({} time-steps)...'.format(e, seq_len))

        for t in range(seq_len):

            # waits until one of the keys is pressed if not advancing episode
            while not advance_episode and not advance_time:
                process_keys()
                pygame.event.pump()

            advance_time = False

            # hides the display when advancing an episode
            if advance_episode:
                pygame.display.iconify()
            else:
                pygame.event.post(pygame.event.Event(pygame.ACTIVEEVENT, {}))

            # gets state-action pair
            old_s = s_seq[t]
            a = a_seq[t]

            # steps the environment, gets next state
            obs, r, done, _ = env.step(a)
            s = helper.get_state_from_observation(obs, r, done)
            r = helper.get_reward(old_s, a, r, s, done)
            helper.update_stats(e, t, old_obs, obs, old_s, a, r, s)

            # checks possible synchronization errors
            if done and t != seq_len - 1:
                raise ValueError(
                    'Environment ended at {}, before tracked behavior which ended at: {}'.format(t, seq_len - 1))
            if t == seq_len - 1 and not done:
                raise ValueError('Environment did not end at {} like it was supposed to'.format(t))
            if not done and s != s_seq[t + 1]:
                raise ValueError('Environment state {} does not match tracked state {}'.format(s, s_seq[t + 1]))

            if not advance_episode:
                obs_vec = get_features_from_index(s, feats_nbins)

                # prints information
                clean_console()
                print('Episode: {}, time-step: {}'.format(e, t))
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

            old_obs = obs
