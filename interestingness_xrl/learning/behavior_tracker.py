__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
from os.path import join, exists
from interestingness_xrl.learning import write_table_csv, read_table_csv, convert_table_to_array, convert_array_to_table

S_A_TABLE_FILE_NAME = 'seq-a-table.csv'
S_S_TABLE_FILE_NAME = 'seq-s-table.csv'


class BehaviorTracker(object):

    def __init__(self, num_episodes):
        self.s_a = [[]]
        """ Corresponds to the sequence of actions performed by the agent at each time-step. """

        self.s_s = [[]]
        """ Corresponds to the sequence of states visited/observed by the agent at each time-step. """

        self.num_episodes = num_episodes
        self._cur_episode = 0

        self.reset()

    def reset(self):
        """
        Resets the tracker by cleaning the state and action trajectories.
        :return:
        """
        self.s_a = [[] for _ in range(self.num_episodes)]
        self.s_s = [[] for _ in range(self.num_episodes)]
        self._cur_episode = 0

    def new_episode(self):
        """
        Signals the tracker that a new episode has started.
        :return:
        """
        if self._cur_episode < self.num_episodes - 1:
            self._cur_episode += 1

    def add_sample(self, state, action):
        """
        Adds a new state-action pair sample to the tracker
        :param int state: the visited state.
        :param int action: the executed action.
        :return:
        """
        self.s_s[self._cur_episode].append(state)
        self.s_a[self._cur_episode].append(action)

    def save(self, output_dir):
        """
        Saves all the relevant information collected by the tracker.
        :param str output_dir: the path to the directory in which to save the data.
        :return:
        """

        # writes each table to a csv file
        write_table_csv(convert_table_to_array(self.s_a, -1), join(output_dir, S_A_TABLE_FILE_NAME))
        write_table_csv(convert_table_to_array(self.s_s, -1), join(output_dir, S_S_TABLE_FILE_NAME))

    def load(self, output_dir):
        """
        Loads all the relevant information collected by the tracker.
        :param str output_dir: the path to the directory from which to load the data.
        :return:
        """

        if not exists(output_dir):
            return
        self.s_a = convert_array_to_table(read_table_csv(join(output_dir, S_A_TABLE_FILE_NAME), dtype=np.int), -1)
        self.s_s = convert_array_to_table(read_table_csv(join(output_dir, S_S_TABLE_FILE_NAME), dtype=np.int), -1)
        self.num_episodes = self._cur_episode = len(self.s_a)
