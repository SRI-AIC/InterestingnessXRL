__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import os
import sys
from os import makedirs
from os.path import exists, join
from interestingness_xrl.learning import get_features_from_index, read_table_csv
from interestingness_xrl.learning.behavior_tracker import BehaviorTracker
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration
from interestingness_xrl.scenarios import DEFAULT_CONFIG, AgentType, get_agent_output_dir, create_helper, \
    get_observations_output_dir

AGENT_TYPE = AgentType.Learning
# AGENT_TYPE = AgentType.Testing
FPS = 5

if __name__ == '__main__':

    # tries to load agent from results dir
    agent_dir = sys.argv[1] if len(sys.argv) > 1 else get_agent_output_dir(DEFAULT_CONFIG, AGENT_TYPE)
    if not exists(agent_dir):
        raise ValueError('Could not load agent from: {}'.format(agent_dir))

    config_file = join(agent_dir, 'config.json')
    if not exists(config_file):
        raise ValueError('Configuration not found: {}'.format(config_file))
    config = EnvironmentConfiguration.load_json(config_file)

    # creates env helper
    helper = create_helper(config)

    # loads the agent's behavior
    behavior_tracker = BehaviorTracker(0)
    behavior_tracker.load(agent_dir)

    # tries to load recorded episodes from results dir
    rec_episodes_file = join(agent_dir, 'rec_episodes.csv')
    if not exists(rec_episodes_file):
        raise ValueError('Recorded episodes file not found: {}'.format(rec_episodes_file))
    recorded_episodes = read_table_csv(rec_episodes_file, dtype=int)

    if len(recorded_episodes) == 0:
        raise ValueError('Could not find any recorded episodes in: {}.'.format(agent_dir))
    print('{} episodes were recorded'.format(len(recorded_episodes)))

    # checks output dir
    output_dir = sys.argv[2] if len(sys.argv) > 2 else get_observations_output_dir(agent_dir)
    if not exists(output_dir):
        makedirs(output_dir)

    # no window mode, to create the images
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

    # for each sequence
    feats_nbins = helper.get_features_bins()
    for e in recorded_episodes:

        # gets state sequence, convert to observation sequence
        obs_vec_seq = []
        s_seq = behavior_tracker.s_s[e]
        for t in range(len(s_seq)):
            obs_vec = get_features_from_index(s_seq[t], feats_nbins)
            obs_vec_seq.append(obs_vec)

        # saves obs sequence to video file
        file_name = 'episode {}.mp4'.format(e)
        print('Creating \'{}\' from sequence with {} time-steps...'.format(file_name, len(obs_vec_seq)))
        helper.save_features_video(obs_vec_seq, join(output_dir, file_name), FPS)
