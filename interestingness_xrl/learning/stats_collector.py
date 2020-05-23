__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import exists
from interestingness_xrl.learning import write_table_csv, read_table_csv, get_combined_mean

DEF_WIDTH = 6
DEF_HEIGHT = 4
NUM_STAT_TYPES = 8


class StatType(object):
    none = -1
    count = 0
    mean = 1
    std_dev = 2
    max = 3
    min = 4
    sum = 5
    last = 6
    ratio = 7

    @staticmethod
    def get_name(stat_t):
        if stat_t == StatType.count:
            return 'Count'
        if stat_t == StatType.mean:
            return 'Mean'
        if stat_t == StatType.std_dev:
            return 'Std-dev'
        if stat_t == StatType.max:
            return 'Max'
        if stat_t == StatType.min:
            return 'Min'
        if stat_t == StatType.sum:
            return 'Sum'
        if stat_t == StatType.last:
            return 'Last'
        if stat_t == StatType.ratio:
            return 'Ratio'
        return 'Unknown'


class StatsCollector(object):
    """
    Represents a collection of statistical variables for which samples are collected during a series of trials.
    """

    def __init__(self):
        """
        Creates a new collector.
        """
        self._vars = {}
        self._counts = {}
        self._stat_types = {}

    def reset(self):
        """
        Resets all counters and clears all collected samples.
        :return:
        """
        # create new variables
        for name in self._vars.keys():
            num_trials, max_samples = self._vars[name].shape
            self.add_sample(name, num_trials, max_samples)

    def add_variable(self, name, num_trials, max_samples, stat_t=StatType.mean):
        """
        Adds a variable whose samples are to be collected during some number of trials.
        :param str name: the name of the variable.
        :param int num_trials: the number of trials during which the variable is going to be sampled.
        :param int max_samples: the maximum number of samples that can be collected during a trial.
        :param int stat_t: the type of statistic we are interested in gathering from this variable at the end of trials.
        """
        self._vars[name] = np.full((num_trials, max_samples), np.nan)
        self._counts[name] = np.zeros(num_trials, dtype=np.int)
        self._stat_types[name] = stat_t

    def all_variables(self):
        """
        Enumerates all variable names stored in this collector.
        :return:
        """
        for var_name in self._vars.keys():
            yield var_name

    def get_type(self, name):
        """
        Gets the type of statistic collected for the given variable.
        :param str name: the name of the variable.
        :rtype: int
        :return: the type of statistic collected for the given variable.
        """
        return self._stat_types[name] if name in self._vars else StatType.none

    def add_sample(self, name, t, val):
        """
        Updates the given statistic by adding in a sample value.
        :param str name: the name of the variable for which we want to add a sample.
        :param int t: the trial number in which the sample was observed.
        :param float val: the new sample to be added.
        :return:
        """
        if name not in self._vars:
            return

        self._vars[name][t][self._counts[name][t]] = val
        self._counts[name][t] += 1

    def get_most_recent_sample(self, name, t):
        """
        Gets the most recent sample added to the given variable.
        :param str name: the name of the variable for which we want to retrieve the most recent sample.
        :param int t: the trial number in which the sample was observed.
        :rtype: float
        :return: the most recent sample added to the given variable.
        """
        return self._vars[name][t][self._counts[name][t] - 1] \
            if name in self._counts and self._counts[name][t] > 0 else np.nan

    def get_trial_count(self, name, t):
        """
        Gets the number of samples collected for the given trial.
        :param str name: the name of the variable for which we want to retrieve the trial count.
        :param int t: the trial number for which we want to retrieve the count.
        :rtype: int
        :return: the number of samples collected for the given trial, or -1 if the given variable name is not in the collection.
        """
        return self._counts[name][t] if name in self._counts else -1

    def get_mean(self, name, t, stat_t=-1):
        """
        Calculates the mean statistic across trials for the given variable.
        :param str name: the name of the variable for which we want the statistic.
        :param int t: the trial number until which to get the statistic.
        :param int stat_t: the type of statistic to be calculated for the variable. If -1, the default statistic is used.
        :rtype: tuple
        :return: (avg, std_dev) a tuple containing the mean statistic and respective standard deviation across trials.
        """
        if t == 0:
            return 0., 0.
        trials_stats = self.get_trials_stats(name, t, stat_t)
        return np.mean(trials_stats), np.std(trials_stats)

    @staticmethod
    def get_stat(data, stat_t):
        """
        Gets a statistic for the given data.
        :param np.ndarray data: the data to get the statistic for.
        :param int stat_t: the type of statistic to be calculated.
        :rtype: float
        :return: the statistic calculated for the given data.
        """
        data_len = len(~np.isnan(data))
        if data_len == 0:
            return np.nan

        if stat_t == StatType.count:
            return data_len
        if stat_t == StatType.mean:
            return np.nanmean(data)
        if stat_t == StatType.std_dev:
            return np.nanstd(data)
        if stat_t == StatType.max:
            return np.nanmax(data)
        if stat_t == StatType.min:
            return np.nanmin(data)
        if stat_t == StatType.sum:
            return np.nansum(data)
        if stat_t == StatType.last:
            return data[data_len - 1]
        if stat_t == StatType.ratio:
            # ratio between the positive samples (count) and num samples
            return np.count_nonzero(data[:data_len]) / data_len

        return np.nan

    def get_trials_stats(self, name, t, stat_t=-1):
        """
        Calculates a statistic for every trial for the given variable.
        :param str name: the name of the variable for which we want the statistic.
        :param int t: the trial number until which to get the statistic.
        :param int stat_t: the type of statistic to be calculated for the variable. If -1, the default statistic is used.
        :rtype: np.ndarray
        :return: an array containing the required statistic for each trial.
        """
        if name not in self._vars:
            return np.full(t, np.nan)

        stat_t = self._stat_types[name] if stat_t == -1 else stat_t
        if stat_t == StatType.count:
            return self._counts[name][:t]
        if stat_t == StatType.mean:
            return np.nanmean(self._vars[name][:t], axis=1)
        if stat_t == StatType.std_dev:
            return np.nanstd(self._vars[name][:t], axis=1)
        if stat_t == StatType.max:
            return np.nanmax(self._vars[name][:t], axis=1)
        if stat_t == StatType.min:
            return np.nanmin(self._vars[name][:t], axis=1)
        if stat_t == StatType.sum:
            return np.nansum(self._vars[name][:t], axis=1)
        if stat_t == StatType.last:
            return [self._vars[name][i][self._counts[name][i] - 1] for i in range(t)]
        if stat_t == StatType.ratio:
            # ratio between the positive samples (count) and num samples
            pos_count = [np.count_nonzero(self._vars[name][i][:self._counts[name][i]]) for i in range(t)]
            return np.true_divide(pos_count, self._counts[name][:t])

        return np.full(t, np.nan)

    @staticmethod
    def get_mean_trials_stats(stats_collectors, name, t, stat_t=-1):
        """
        Calculates the mean of a statistic for every trial of the given variable, across multiple collections.
        :param list stats_collectors: a list of StatsCollector objects, assumed to have the same number of trials.
        :param str name: the name of the variable for which we want the mean statistic.
        :param int t: the trial number until which to get the statistic.
        :param int stat_t: the type of statistic to be calculated for the variable. If -1, the default statistic is used.
        :rtype: (np.ndarray, np.ndarray, np.ndarray)
        :return: a tuple (means, stds, counts) containing the mean and standard deviation of the required statistic for
        each trial, across all collectors.
        """
        stat_t = stats_collectors[0]._stat_types[name] if stat_t == -1 else stat_t
        if stat_t == StatType.mean:
            # if stat itself is the mean, we need to collect the combined mean for each trial
            var_means = np.array([collector.get_trials_stats(
                name, t, StatType.mean) for collector in stats_collectors])
            var_stds = np.array([collector.get_trials_stats(
                name, t, StatType.std_dev) for collector in stats_collectors])
            var_counts = np.array([collector.get_trials_stats(
                name, t, StatType.count) for collector in stats_collectors])

            var_trials_stats = []
            for t in range(var_means.shape[1]):
                var_trials_stats.append(get_combined_mean(var_means[:, t], var_stds[:, t], var_counts[:, t]))
            var_trials_stats = np.array(var_trials_stats)
            return var_trials_stats[:, 0], var_trials_stats[:, 1], var_trials_stats[:, 2]

        # otherwise just calculate means for each trial
        var_trials_stats = np.array([collector.get_trials_stats(name, t, stat_t) for collector in stats_collectors])
        return (var_trials_stats.mean(axis=0),
                var_trials_stats.std(axis=0),
                np.full(var_trials_stats.shape[1], var_trials_stats.shape[0]))

    def get_across_trials_stats(self, name, t, stat_t=-1):
        """
        Gets all statistics of the given variable when considering the stat for each trial, i.e., the across-trial mean,
        std dev, max, min, ...
        :param str name: the name of the variable for which we want the statistic.
        :param int t: the trial number until which to get the statistic.
        :param int stat_t: the type of statistic to be calculated for the variable for each trial. If -1, the default
        statistic is used.
        :rtype: np.ndarray
        :return: an array containing the several across-trial statistics for the variable, ordered by stat type index.
        """
        trials_stats = self.get_trials_stats(name, t, stat_t)
        across_trials_stats = np.zeros(NUM_STAT_TYPES)
        for stat_idx in range(NUM_STAT_TYPES):
            across_trials_stats[stat_idx] = self.get_stat(trials_stats, stat_idx)
        return across_trials_stats

    def save(self, name, t, file_path, binary=False, compressed=False, delimiter=','):
        """
        Saves all the samples collected for the given variable in a text or binary file.
        :param str name: the name of the variable for which we want the statistic.
        :param int t: the trial number until which to save the samples.
        :param str file_path: the path to the file.
        :param bool binary: whether to save a binary file. If False, a CSV text file will be used.
        :param bool compressed: whether to save a compressed binary file. Only works if binary is True.
        :param str delimiter: the delimiter for the fields in the CSV file, if Binary is False.
        :return:
        """
        if name in self._vars:
            data = self._vars[name][:t]
            if binary:
                if compressed:
                    np.savez_compressed('{}.npz'.format(file_path), a=data)
                else:
                    np.save('{}.npy'.format(file_path), data)
            else:
                col_names = ['Trial {}'.format(_t) for _t in range(t)]
                write_table_csv(data.T, '{}.csv'.format(file_path), delimiter, '%s', col_names)

    def load(self, name, file_path, stat_t=StatType.mean, binary=False, delimiter=','):
        """
        Loads the data regarding a variable previously-saved by a stats collector from a file. Adds the variable sample
        data per trial to the collection of variables in this collector.
        :param str name: the name of the variable that we want to load.
        :param str file_path: the path to the file.
        :param int stat_t: the type of statistic of the variable to be loaded.
        :param bool binary: whether to load a binary file. If False, a CSV text file will be loaded.
        :param str delimiter: the delimiter for the fields in the CSV file, if binary is False.
        :rtype: bool
        :return: whether the data was successfully loaded.
        """
        # tries to load files in different formats
        data = None
        if binary:
            if exists('{}.npz'.format(file_path)):
                data = np.load('{}.npz'.format(file_path))['a']
            elif exists('{}.npy'.format(file_path)):
                data = np.load('{}.npy'.format(file_path))
        elif exists('{}.csv'.format(file_path)):
            data = read_table_csv('{}.csv'.format(file_path), delimiter, has_header=True).T
        if data is None:
            return False

        # sets variable data if loaded correctly
        self._vars[name] = data
        self._counts[name] = np.count_nonzero(~np.isnan(data), axis=1)
        self._stat_types[name] = stat_t
        return True

    def save_across_trials_image(
            self, name, t, file_path, stat_t=-1, x_label='Trial', width=DEF_WIDTH, height=DEF_HEIGHT):
        """
        Saves an image plotting the evolution of the statistic associated with the given variable across trials.
        :param str name: the name of the variable for which we want to save the image.
        :param int t: the trial number until which to get the statistic.
        :param str file_path: the path to the image file.
        :param int stat_t: the type of statistic to be plotted for the variable. If -1, the default statistic is used.
        :param str x_label: the label of the X-axis
        :param int width: the width of the image.
        :param int height: the height of the image.
        :return:
        """
        if name not in self._vars:
            return

        # gets array with stats for each trial
        stat_t = self._stat_types[name] if stat_t == -1 else stat_t
        data = self.get_trials_stats(name, t, stat_t)
        x = range(1, len(data) + 1)

        # creates and prints plot to image file
        plt.figure(figsize=(width, height))
        plt.plot(x, data)

        self._format_plot(x.stop, x_label, '{} {}'.format(StatType.get_name(stat_t), name))
        plt.savefig(file_path)
        plt.close()

    def save_mean_image(self, name, t, file_path, x_label='Sample', width=DEF_WIDTH, height=DEF_HEIGHT):
        """
        Saves an image plotting the evolution of the mean value of the given variable across trials.
        :param str name: the name of the variable for which we want to save the image.
        :param int t: the trial number until which to get the statistic.
        :param str file_path: the path to the image file.
        :param int width: the width of the image.
        :param int height: the height of the image.
        :param str x_label: the label to appear in the x-axis.
        :return:
        """
        if name not in self._vars:
            return

        # converts data to data frame
        data = self._vars[name][:t]
        df = pd.DataFrame(data)

        # gets means and std errors along trials
        means = df.mean(axis=0)
        counts = df.count(axis=0)
        std_errors = df.std(axis=0) / np.sqrt(counts)
        mins = means - std_errors
        maxs = means + std_errors
        x = range(1, data.shape[1] + 1)

        # creates and prints plot to image file
        plt.figure(figsize=(width, height))
        plt.plot(x, means)
        plt.fill_between(x, mins, maxs, alpha=0.5)

        self._format_plot(x.stop, x_label, 'Mean ' + name.lower())
        plt.savefig(file_path)
        plt.close()

    @staticmethod
    def _format_plot(max_x, x_label, y_label):

        plt.xlim(1, max_x)
        plt.xlabel(x_label, fontweight='bold', fontsize=14)
        plt.ylabel(y_label, fontweight='bold', fontsize=14)
