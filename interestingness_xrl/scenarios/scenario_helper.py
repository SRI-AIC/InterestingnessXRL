__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
import pygame
from os import makedirs
from shutil import rmtree
from os.path import join, exists
from abc import abstractmethod
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration
from interestingness_xrl.learning import get_discretized_index
from interestingness_xrl.learning.stats_collector import StatsCollector, StatType
from interestingness_xrl.learning.explorations import SoftMaxExploration, EpsilonGreedyExploration
from interestingness_xrl.learning.agents import QValueBasedAgent
from interestingness_xrl.util import record_video, print_line

CSV_DELIMITER = ','

REWARD_VAR = 'Reward'
EXPLORATION_VAR = 'Exploration'
STATE_VALUE_VAR = 'State value'
STATE_COUNT_VAR = 'State count'
TIME_STEPS_VAR = 'Time-steps'

ANY_FEATURE_IDX = -1  # special, used to represent any other feature


class ScenarioHelper(object):
    """
    Represents a set of helper methods for learning and explanation analysis of a type of environments/games.
    """

    def __init__(self, config):
        """
        Creates a new environment helper according to the given configuration.
        :param EnvironmentConfiguration config: the configuration with all necessary parametrization.
        """
        self.config = config

        # to be set if we want to collect agent stats
        self.agent = None

        # default stats to be collected from the agent
        self.stats_collector = StatsCollector()
        self._add_variable(TIME_STEPS_VAR, StatType.last)
        self._add_variable(REWARD_VAR, StatType.sum)
        self._add_variable(EXPLORATION_VAR, StatType.last)
        self._add_variable(STATE_VALUE_VAR, StatType.mean)
        self._add_variable(STATE_COUNT_VAR, StatType.mean)

    @abstractmethod
    def register_gym_environment(self, env_id='my-env-v0', display_screen=False, fps=30, show_score_bar=False):
        """
        Registers an OpenAI gym environment in the system (if needed).
        :param str env_id: the environment id to be registered.
        :param bool display_screen: whether to display the environment screen.
        :param int fps: the number of frames-per-second that the environment should be recorded.
        :param bool show_score_bar: whether to show the score bar at the top of the screen.
        :return:
        """
        pass

    @abstractmethod
    def get_agent_cell_location(self, obs):
        """
        Gets the (col, row) coordinates of the agent's cell in the environment, according to the given observation.
        :param np.ndarray obs: the observation matrix / game state provided to the agent.
        :return int, int: the (column, row) coordinates of the cell where the agent is located.
        """
        pass

    def get_cell_coordinates(self, col, row):
        """
        Gets the game (x, y) coordinates of the given cell.
        :param int col: the column coordinate of the cell.
        :param int row: the row coordinate of the cell.
        :return int, int: the (x, y) game coordinates of the required cell.
        """
        return col * self.config.cell_size[0], row * self.config.cell_size[1]

    def get_state_from_observation(self, obs, rwd, done):
        """
        Discretizes the given observation by calculating the state index.
        :param np.ndarray obs: the observation matrix / game state provided to the agent.
        :param float rwd: the reward resulting from the update.
        :param bool done: whether the underlying world engine reported that the gym environment has ended.
        :return int: the index of the discretized state corresponding to the given observation.
        """

        # tests for terminal / absorbing state
        if self.is_terminal_state(obs, rwd, done):
            return self.get_terminal_state()

        obs_vec = self.get_features_from_observation(obs)
        feats_nbins = self.get_features_bins()

        # gets discretized index
        return get_discretized_index(obs_vec, feats_nbins)

    def get_state_from_features(self, obs_vec):
        """
        Gets the state index corresponding to the given set of discretized features.
        :param array_like obs_vec: the discretized features for which to calculate the state index.
        :return int: the state index corresponding to the discretized features.
        """
        return get_discretized_index(obs_vec, self.get_features_bins())

    @abstractmethod
    def get_features_from_observation(self, obs, agent_x=-1, agent_y=-1):
        """
        Transforms the given observation of the environment into a set of discretized state features.
        :param np.ndarray obs: the observation matrix / game state provided to the agent.
        :param int agent_x: the X location of the agent in the environment. If -1, it has to be collected from the observation.
        :param int agent_y: the Y location of the agent in the environment. If -1, it has to be collected from the observation.
        :return array: an array containing the discretized state features.
        """
        pass

    @abstractmethod
    def get_features_bins(self):
        """
        Gets an array containing the number of bins (elements) for each state feature.
        :return array: an array containing the number of bins (elements) for each state feature.
        """
        pass

    @abstractmethod
    def get_terminal_state(self):
        """
        Gets a generic terminal state used for this environment.
        :return int: the state index corresponding to the terminal state.
        """
        pass

    def is_terminal_state(self, obs, rwd, done):
        """
        Determines whether the given gym update corresponds to a terminal state in this environment.
        :param ndarray obs: the observation provided to the agent.
        :param float rwd: the reward resulting from the update.
        :param bool done: whether the underlying world engine reported that the gym environment has ended.
        :return bool: whether the given gym update corresponds to a terminal state.
        """
        # default is just to return according to done
        return done

    def get_reward(self, s, a, rwd, ns, done):
        """
        Computes the 'learning' reward given all transition information, including the original game's reward.
        :param int s: the previous state, from which the action was executed.
        :param int a: the action executed by the agent.
        :param int ns: the next state, resulting from the update.
        :param float rwd: the reward resulting from the update.
        :param bool done: whether the underlying world engine reported that the gym environment has ended.
        :return:
        """
        # default is to simply return the game's original reward
        return rwd

    @abstractmethod
    def get_feature_label(self, obs_feat_idx, obs_feat_val):
        """
        Gets the label corresponding to the given feature.
        :param int obs_feat_idx: the discretized feature index.
        :param int obs_feat_val: the discretized feature value.
        :return str the label corresponding to the given feature:
        """
        pass

    @abstractmethod
    def get_features_labels(self, obs_vec, short=False):
        """
        Gets a description for each of the given state features to the standard output.
        :param np.ndarray obs_vec: an array of discretized features, i.e., containing the indexes to the features' elements.
        :param bool short: whether to return a shortened version of the labels.
        :return list: a list containing a textual description for each of the given state features.
        """
        pass

    @abstractmethod
    def print_features(self, obs_vec, columns=False):
        """
        Prints a description of the given state features to the standard output.
        :param np.ndarray obs_vec: an array of discretized features, i.e., containing the indexes to the features' elements.
        :param bool columns: whether to print features organized in columns.
        :return:
        """
        pass

    @abstractmethod
    def get_transaction(self, obs_vec, short=False):
        """
        Converts the given set of discretized features into an item-set-like transaction to be used in pattern mining.
        :param np.ndarray obs_vec: an array of discretized features, i.e., containing the indexes to the features' elements.
        :param bool short: whether to return a shortened version of the labels.
        :return:
        """
        pass

    def get_feature_dissimilarity(self, obs_vec1, obs_vec2):
        """
        Gets the dissimilarity (distance) between the two given observation-feature vectors.
        :param np.ndarray obs_vec1: an array of discretized features, i.e., containing the indexes to the features' elements.
        :param np.ndarray obs_vec2: an array of discretized features, i.e., containing the indexes to the features' elements.
        :rtype: float
        :return: a number in [0,1], indicating how different the two observation-features are.
        """
        # default returns division of intersection and union of features
        obs_vec1_feats = set(['{}:{}'.format(i, feat) for i, feat in enumerate(obs_vec1)])
        obs_vec2_feats = set(['{}:{}'.format(i, feat) for i, feat in enumerate(obs_vec2)])
        union = obs_vec1_feats | obs_vec2_feats
        inter = obs_vec1_feats & obs_vec2_feats
        return 1. - len(inter) / len(union)

    def get_observation_dissimilarity(self, obs1, obs2):
        """
        Gets the dissimilarity (distance) between the two given observations.
        :param np.ndarray obs1: the first observation as provided to the agent.
        :param np.ndarray obs2: the second observation as provided to the agent.
        :rtype: float
        :return: a number in [0,1], indicating how different the two observations are.
        """
        # default is the norm/Euclidean distance between the arrays
        return np.linalg.norm(obs1 - obs2)

    @abstractmethod
    def act_reactive(self, s, rng):
        """
        Chooses an action to be executed by the agent. It corresponds to a known (not learned) strategy for this
        environment.
        :param int s: the state that the agent is going to react to.
        :param np.random.RandomState rng: the random number generator to be used to select actions stochastically.
        :rtype: int
        :return: the index of the action to be executed by the agent.
        """
        pass

    @abstractmethod
    def get_known_goal_states(self):
        """
        Gets the expected / known goal states associated with this environment.
        :return array: an int array containing the known goal states' indexes.
        """
        pass

    @abstractmethod
    def get_known_feature_action_assocs(self):
        """
        Gets the known associations between action execution and feature states in this environment, i.e., the actions
        that are expected to be executed if some feature is present in the agent's state.
        :return array: an array containing the known feature-action associations in the format (feature, value, action).
        """
        pass

    def _add_variable(self, name, stat_t):
        """
        Adds a new variable to be tracked.
        :param str name: the name of the variable
        :param int stat_t: the type of statistic we are interested in gathering from this variable at the end of episodes.
        :return:
        """
        self.stats_collector.add_variable(name, self.config.num_episodes, self.config.max_steps_per_episode, stat_t)

    def update_stats(self, e, t, obs, n_obs, s, a, r, ns):
        """
        Updates statistics collected for this environment based on the given agent update information.
        :param int e: the episode number in which the transition was observed.
        :param int t: the time-step in which the transition was observed.
        :param ndarray obs: the previous observation provided to the agent.
        :param ndarray n_obs: the current observation provided to the agent.
        :param int s: the state at time-step t (from which the agent departed).
        :param int a: the action executed by the agent at time-step t.
        :param float r: the reward that the agent has received.
        :param int ns: the state that resulted from the action execution.
        :return:
        """
        # gets info from the agent if available
        self.stats_collector.add_sample(REWARD_VAR, e, r)
        if self.agent is None:
            return

        # gets exploration variable if available
        explore_var = self.agent.exploration_strategy.temp \
            if isinstance(self.agent.exploration_strategy, SoftMaxExploration) \
            else self.agent.exploration_strategy.eps \
            if isinstance(self.agent.exploration_strategy, EpsilonGreedyExploration) else 0

        # gets state value variable if available
        value_var = np.max(self.agent.q[s]) if isinstance(self.agent, QValueBasedAgent) else 0

        # collects agent-related stats
        self.stats_collector.add_sample(TIME_STEPS_VAR, e, t)
        self.stats_collector.add_sample(EXPLORATION_VAR, e, explore_var)
        self.stats_collector.add_sample(STATE_VALUE_VAR, e, value_var)
        self.stats_collector.add_sample(STATE_COUNT_VAR, e, self.agent.c_s[s])

    @abstractmethod
    def update_stats_episode(self, e, path=None):
        """
        Updates statistics collected for one episode.
        :param int e: the number of the episode.
        :param str path: the path (directory or file) in which to save the statistical information.
        :return:
        """
        pass

    def save_stats(self, path, clear=True, img_format='pdf'):
        """
        Finalizes all collected statistics for this environment.
        :param str path: the path (directory or file) in which to save the statistical information.
        :param bool clear: whether to delete the contents of the results directory.
        :param str img_format: the image format (extension) for the image files.
        :return:
        """
        # checks dir
        if not exists(path):
            makedirs(path)
        elif clear:
            rmtree(path)
            makedirs(path)

        # prints all stats individually to an image file, CSV and binary files
        t = self.config.num_episodes
        for var_name in self.stats_collector.all_variables():
            self.stats_collector.save_mean_image(
                var_name, t, join(path, 'avg {}.{}'.format(var_name.lower(), img_format)), 'Time-step')
            self.stats_collector.save_across_trials_image(
                var_name, t, join(path, 'evo {}.{}'.format(var_name.lower(), img_format)), x_label='Episode')
            self.stats_collector.save(var_name, t, join(path, var_name.lower()), False)
            self.stats_collector.save(var_name, t, join(path, var_name.lower()), True, True)

    def load_stats(self, path):
        """
        Tries to load statistics collected previously for all variables of this environment.
        :param str path: the path (directory or file) from which to load the statistical information.
        :return:
        """
        # tries to load all variables from file (first tries loading binary files over CSV text)
        for var_name in self.stats_collector.all_variables():
            file_path = join(path, var_name.lower())
            if not self.stats_collector.load(var_name, file_path, self.stats_collector.get_type(var_name), True) and \
                    not self.stats_collector.load(var_name, file_path, self.stats_collector.get_type(var_name), False):
                raise ValueError('Could not load variable {} from: {}.'.format(var_name, file_path))

    def _print_stats(self, e, selected_var_names, file=None):
        """
        Prints the current stats of a selected number of variables to the screen.
        :param int e: the number of the episode.
        :param list selected_var_names: a list of the names of the variables for which to print the statistics.
        :param stream file: the file on which to save the message line.
        :return:
        """
        for var_name in selected_var_names:
            avg, std = self.stats_collector.get_mean(var_name, e)
            print_line('Avg {}: {:.2f} ± {:.2f}'.format(var_name.lower(), avg, std), file)

    @abstractmethod
    def save_state_features(self, out_file, delimiter=CSV_DELIMITER):
        """
        Saves a description of all states in terms of the elements in each corresponding feature to a CSV file.
        :param str out_file: the path to the output CSV file in which to save the states' descriptions.
        :param str delimiter: the CSV field delimiter.
        :return:
        """
        pass

    def save_features_image(self, obs_vec, out_file):
        """
        Saves the given observation features into an image file, i.e., the features are saved in a visual form.
        :param np.ndarray obs_vec: an array of discretized features, i.e., containing the indexes to the features' elements.
        :param str out_file: the path to the output file in which to save the features image.
        :return:
        """
        # default is to get features image and save to file
        surf = self.get_features_image(obs_vec)
        pygame.image.save(surf, out_file)

    @abstractmethod
    def get_features_image(self, obs_vec):
        """
        Converts the given observation into an image representation.
        :param np.ndarray obs_vec: an array of discretized features, i.e., containing the indexes to the features' elements.
        :return Surface: a surface object containing an image of the observation.
        """
        pass

    def save_features_video(self, obs_vec_seq, out_file, fps=30):
        """
        Converts a given sequence of observations into a video file.
        :param list obs_vec_seq: a sequence of arrays of discretized features, i.e., containing the indexes to the features' elements.
        :param str out_file: the path to the video file in which to save the observation features sequence.
        :param int fps: the number of frames-per-second to be recorded in the video.
        """

        # converts each observation vector to a feature image
        frame_buffer = []
        for obs_vec in obs_vec_seq:
            surf = self.get_features_image(obs_vec)
            frame = pygame.surfarray.array3d(surf).astype(np.uint8)
            frame = np.fliplr(np.rot90(frame, 3))
            frame_buffer.append(frame)

        # records frame buffer to video file
        record_video(frame_buffer, out_file, fps)
