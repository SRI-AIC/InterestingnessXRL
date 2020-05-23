__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import jsonpickle
from interestingness_xrl.util import dict_to_list, list_to_dict


class EnvironmentConfiguration(object):
    """
    Represents a configuration of tests / learning simulations.
    """

    def __init__(self, name, num_states, actions, rewards, gym_env_id,
                 max_steps_per_episode=300, num_episodes=1000, num_recorded_videos=10,
                 seed=0, max_temp=20, min_temp=0.05, discount=.9, learn_rate=.3, initial_q_value=0.,
                 cell_size=(20, 20), env_size=(10, 10), img_tile_size=(20, 20)):
        """
        Creates a new configuration with the given parameters.
        :param str name: the name of the configuration.
        :param OrderedDict actions: the actions available for the agent in a 'action_name : [keyboard_codes]' fashion.
        :param dict rewards: the reward function in an 'element_name : value' fashion.
        :param str gym_env_id: the name identifier for the gym environment.
        :param int max_steps_per_episode: the maximum number of steps in one episode.
        :param int num_episodes: the number of episodes used to train/test the agent.
        :param int num_recorded_videos: the number of videos to record during the agent run.
        :param int seed: the seed used for the random number generator used by the agent.
        :param float max_temp: the maximum temperature of the Soft-max action-selection strategy (start of training).
        :param float min_temp: the minimum temperature of the Soft-max action-selection strategy (end of training).
        :param float discount: the discount factor in [0, 1] (how important are the future rewards?).
        :param float learn_rate: the agent's learning rate (the weight associated to a new sample during learning).
        :param float initial_q_value: the value used to initialize the q-function (e.g., for optimistic initialization).
        :param tuple cell_size: the (width, height) of an agent cell in this environment.
        :param tuple env_size: the (num_cols, num_rows) in the environment.
        :param tuple img_tile_size: the (width, height) size for observation tiles.
        """
        self.name = name
        self.num_states = num_states
        self.num_actions = len(actions)
        self.actions = actions
        self.rewards = rewards
        self.gym_env_id = gym_env_id
        self.num_episodes = num_episodes
        self.num_recorded_videos = num_recorded_videos
        self.seed = seed
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.discount = discount
        self.learn_rate = learn_rate
        self.initial_q_value = initial_q_value
        self.max_steps_per_episode = max_steps_per_episode
        self.cell_size = cell_size
        self.env_size = env_size
        self.obs_tile_size = img_tile_size

    def get_action_names(self):
        """
        Gets a list with the names of the actions available to the agent.
        :return list: a list with the names of the actions available to the agent, in the order they were added to the
        actions dictionary.
        """
        return list(self.actions.keys())

    def save_json(self, json_file_path):
        """
        Saves a text file representing this configuration in a JSON format.
        :param str json_file_path: the path to the JSON file in which to save this configuration.
        :return:
        """
        # transform ordered dicts to list for a more clear json
        actions = self.actions
        self.actions = dict_to_list(actions)

        jsonpickle.set_preferred_backend('json')
        jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)
        with open(json_file_path, 'w') as json_file:
            json_str = jsonpickle.encode(self)
            json_file.write(json_str)

        self.actions = actions

    @staticmethod
    def load_json(json_file_path):
        """
        Loads a configuration object from the given JSON formatted file.
        :param str json_file_path: the path to the JSON file from which to load a configuration.
        :rtype: EnvironmentConfiguration
        :return: the configuration object stored in the given JSON file.
        """
        with open(json_file_path) as json_file:
            config = jsonpickle.decode(json_file.read())
            config._after_loaded_json()

        return config

    def _after_loaded_json(self):
        # convert lists to ordered dictionaries
        self.actions = list_to_dict(self.actions)


class AnalysisConfiguration(object):
    """
    Represents a configuration for explanation analysis of learning simulation environments.
    """

    def __init__(self, min_count=10, certain_trans_max_disp=0.001, trans_min_states=5,
                 uncertain_trans_min_disp=0.2, rwd_outlier_stds=2, freq_min_state_count=3000, infreq_max_state_count=3,
                 min_feat_set_count=3000, assoc_min_feat_set_jacc=0.75, assoc_min_feat_rule_conf=0.9,
                 no_assoc_max_feat_set_jacc=0.04, certain_exec_max_disp=0.3, uncertain_exec_min_disp=0.9,
                 value_outlier_stds=1.6, pred_error_outlier_stds=2, max_time_step=0.5, val_diff_var_outlier_stds=2,
                 action_jsd_threshold=0.05):
        # general
        self.min_count = min_count

        # transition
        self.trans_min_states = trans_min_states
        self.certain_trans_max_disp = certain_trans_max_disp
        self.uncertain_trans_min_disp = uncertain_trans_min_disp

        # reward
        self.rwd_outlier_stds = rwd_outlier_stds

        # state frequency
        self.freq_min_state_count = freq_min_state_count
        self.infreq_max_state_count = infreq_max_state_count
        self.min_feat_set_count = min_feat_set_count
        self.assoc_min_feat_set_jacc = assoc_min_feat_set_jacc
        self.assoc_min_feat_rule_conf = assoc_min_feat_rule_conf
        self.no_assoc_max_feat_set_jacc = no_assoc_max_feat_set_jacc

        # state-action frequency
        self.certain_exec_max_disp = certain_exec_max_disp
        self.uncertain_exec_min_disp = uncertain_exec_min_disp

        # value
        self.value_outlier_stds = value_outlier_stds
        self.pred_error_outlier_stds = pred_error_outlier_stds

        # recency
        self.max_time_step = max_time_step

        # transition value
        self.val_diff_var_outlier_stds = val_diff_var_outlier_stds

        # contradictions
        self.action_jsd_threshold = action_jsd_threshold

    def save_json(self, json_file_path):
        """
        Saves a text file representing this configuration in a JSON format.
        :param str json_file_path: the path to the JSON file in which to save this configuration.
        :return:
        """
        jsonpickle.set_preferred_backend('json')
        jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)
        with open(json_file_path, 'w') as json_file:
            json_str = jsonpickle.encode(self)
            json_file.write(json_str)

    @staticmethod
    def load_json(json_file_path):
        """
        Loads an analysis object from the given JSON formatted file.
        :param str json_file_path: the path to the JSON file from which to load a configuration.
        :return AnalysisConfig: the configuration object stored in the given JSON file.
        """
        with open(json_file_path) as json_file:
            return jsonpickle.decode(json_file.read())
