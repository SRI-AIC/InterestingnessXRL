__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from os import makedirs, pardir
from os.path import join, exists
from shutil import rmtree
from matplotlib.ticker import FuncFormatter
from collections import OrderedDict
from interestingness_xrl.util import print_line
from interestingness_xrl.learning import get_combined_mean
from interestingness_xrl.learning.stats_collector import NUM_STAT_TYPES, StatType, StatsCollector
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration
from interestingness_xrl.scenarios import get_agent_output_dir, create_helper, AgentType, DATA_ORGANIZER_CONFIGS, \
    get_analysis_output_dir
from interestingness_xrl.explainability.analysis.full_analysis import FullAnalysis
from scipy.signal import savgol_filter

POLY_ORDER = 3
WINDOW_LENGTH = 71

# DEF_AGENT_TYPE = AgentType.Learning
DEF_AGENT_TYPE = AgentType.Testing
CLEAN_DIR = True
COLOR_SET_NAME = 'Set1'

STAT_NAMES = [StatType.get_name(idx) for idx in range(NUM_STAT_TYPES)]


def clear_dir(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)
    elif CLEAN_DIR:
        rmtree(dir_path)
        makedirs(dir_path)


def print_mean_comparison(_output_dir, _stats, _var_names):
    # creates comparison files for each variable
    for _var_name in _var_names:
        print_variable_mean_comparison(_output_dir, _stats, _var_name)


def print_variable_mean_comparison(_output_dir, _stats, _var_name):
    print_line('\n==============================================', log_file)
    print_line('Processing comparison of means for \'{}\'...\n'.format(_var_name.lower()), log_file)

    # reorganizes data by stats x agent
    ag_names = list(_stats.keys())
    var_data = np.zeros((3, len(_stats)))
    for _ag_idx, _ag_name in enumerate(ag_names):
        means, stds, counts = _stats[_ag_name][_var_name]
        mean, std, count = get_combined_mean(means, stds, counts)
        var_data[:, _ag_idx] = [mean, std, count]

    # convert variable stats to pandas data-frame
    df = pd.DataFrame(var_data, index=['Mean', 'Std', 'Count'], columns=ag_names)

    # saves to CSV
    file_name = _var_name.lower()
    df.to_csv(join(_output_dir, '{}.csv'.format(file_name)))

    # print to screen
    print_line(str(df), log_file)

    x_axis = np.arange(len(ag_names))
    means = [var_data[0][i] for i in x_axis]
    errors = [var_data[1][i] / np.sqrt(var_data[2][i]) for i in x_axis]
    # errors = [var_data[1][i] for i in x_axis]
    color_map = plt.cm.get_cmap(COLOR_SET_NAME)(x_axis)

    # gets y label adjust for base 10
    exp = get_y_label_adjust_base(max(means) + max(errors))
    y_label = 'Mean {}'.format(_var_name).title()

    # print bar chart of means to image file (may not work well for all variables)
    plt.bar(x_axis, means, yerr=errors, capsize=10, align='center', color=color_map)
    plt.xticks(x_axis, ag_names)
    plt.xlabel('Agent', fontweight='bold', fontsize=14)
    if abs(exp) >= 2:
        y_label = '{} $\\mathregular{{\\left(\\times10^{}\\right)}}$'.format(y_label, exp)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{}'.format(int(x / (10 ** exp)))))
    plt.ylabel(y_label, fontweight='bold', fontsize=14)
    plt.savefig(join(_output_dir, '{}.pdf'.format(file_name)), pad_inches=.02, bbox_inches='tight')
    plt.close()


def print_evo_comparison(_output_dir, _stats, _var_names):
    # creates comparison files for each variable
    for _var_name in _var_names:
        print_variable_evo_comparison(_output_dir, _stats, _var_name)


def smooth_data(y):
    return savgol_filter(y, WINDOW_LENGTH, POLY_ORDER)


def print_variable_evo_comparison(_output_dir, _stats, _var_name):
    print_line('\n==============================================', log_file)
    print_line('Processing comparison of evolution for \'{}\'...\n'.format(_var_name.lower()), log_file)

    # plots and creates data frame from trial data for each variable
    plt.figure()
    ag_names = list(_stats.keys())
    color_map = plt.cm.get_cmap(COLOR_SET_NAME)(np.arange(len(ag_names)))

    var_data = OrderedDict()
    for _ag_idx, _ag_name in enumerate(ag_names):
        means, stds, counts = _stats[_ag_name][_var_name]
        var_data['{} Mean'.format(_ag_name)] = means
        var_data['{} Std'.format(_ag_name)] = stds
        var_data['{} Count'.format(_ag_name)] = counts
        errs = stds / np.sqrt(counts)

        plt.plot(smooth_data(means), label=_ag_name, c=color_map[_ag_idx])
        plt.fill_between(np.arange(len(means)),
                         smooth_data(means - errs),
                         smooth_data(means + errs), color=color_map[_ag_idx], alpha=0.5)

    df = pd.DataFrame(var_data)
    df.to_csv(join(subdir, '{}-trial-data.csv'.format(_var_name.lower())))

    # gets y label adjust for base 10
    exp = get_y_label_adjust_base(df.max().max())
    y_label = _var_name.title()

    # configures axes and legend
    plt.legend(fontsize=16)
    plt.xlabel('Episode', fontweight='bold', fontsize=14)
    if abs(exp) >= 2:
        y_label = '{} $\\mathregular{{\\left(\\times10^{}\\right)}}$'.format(y_label, exp)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{}'.format(int(x / (10 ** exp)))))
    plt.ylabel(y_label, fontweight='bold', fontsize=14)

    plt.xlim(0, df.shape[0])

    # saves figure
    plt.savefig(join(_output_dir, 'evo {}.pdf'.format(_var_name.lower())), pad_inches=.02, bbox_inches='tight')
    plt.close()


def get_y_label_adjust_base(val):
    val = abs(val)
    if 0 < val < 1:
        exp = -1
        while val < 10 ** exp:
            exp -= 1
        return exp + 2
    exp = 1
    while val > 10 ** exp:
        exp += 1
    return exp - 2


def get_agent_data(config, trial_num):
    # loads config from agent dir
    agent_dir = join(pardir, get_agent_output_dir(config, agent_t, trial_num))
    config_file = join(agent_dir, 'config.json')
    if not exists(config_file):
        return None, None
    config = EnvironmentConfiguration.load_json(config_file)

    # creates env helper
    helper = create_helper(config)

    # tries to load full analysis
    analyses_dir = get_analysis_output_dir(agent_dir)
    file_name = join(analyses_dir, 'full-analysis.json')
    analysis = None
    if exists(file_name):
        analysis = FullAnalysis.load_json(file_name)
        analysis.set_helper(helper)

    # tries to load all data
    stats_dir = join(agent_dir, 'results')
    if exists(stats_dir):
        helper.load_stats(stats_dir)
    else:
        helper = None

    return analysis, helper


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='RL Agent performance comparer')
    parser.add_argument('-a', '--agent', help='agent type', type=int, default=DEF_AGENT_TYPE)
    parser.add_argument('-o', '--output', help='directory in which to store results')
    args = parser.parse_args()

    # sets pandas options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.float_format', '{:,.2f}'.format)

    # tries to get agent type
    agent_t = args.agent

    # tries to load the stats for each agent
    analyses = []
    helpers = []
    ag_names = []
    for ag_name, config in DATA_ORGANIZER_CONFIGS.items():
        ag_names.append(ag_name)
        ag_helpers = []
        trial_num = 0
        while True:

            analysis, helper = get_agent_data(config, trial_num)

            # we need at least some performance data
            if helper is None:
                if trial_num == 0:
                    raise ValueError('Helper stats not found for agent \'{}\''.format(ag_name))
                break
            ag_helpers.append(helper)

            # we need analysis data only for testing
            if trial_num == 0:
                if analysis is None and agent_t == AgentType.Testing:
                    raise ValueError('Full analysis not found for agent \'{}\''.format(ag_name))
                elif analysis is not None:
                    analyses.append(analysis)

            trial_num += 1

        helpers.append(ag_helpers)
        print('Loaded stats of {} trials for agent \'{}\' ({}).'.format(trial_num, ag_name, config.name))

    # creates output dir if needed
    output_dir = args.output if args.output is not None else \
        join(pardir, 'results', 'agent_comparer', AgentType.get_name(agent_t))
    clear_dir(output_dir)

    # creates log file
    log_file = open(join(output_dir, 'log.txt'), 'w')
    print()

    # ============================================================
    # PART 1 : collects performance stats across trials/episodes per agent
    subdir = join(output_dir, 'performance')
    clear_dir(subdir)

    stats = OrderedDict()
    var_names = list(helpers[0][0].stats_collector.all_variables())
    for ag_idx, ag_helpers in enumerate(helpers):

        ag_name = ag_names[ag_idx]
        print_line('Collecting stats across {} trials of {} episodes for agent \'{}\'...'.format(
            len(ag_helpers), ag_helpers[0].config.num_episodes, ag_name), log_file)

        # saves / copies configs to file
        ag_helpers[0].config.save_json(join(output_dir, 'config-{}.json'.format(ag_name.lower())))

        # collects stats for all trials
        stats[ag_name] = {}
        stats_collectors = [helper.stats_collector for helper in ag_helpers]
        for var_name in var_names:
            ag_stats = StatsCollector.get_mean_trials_stats(
                stats_collectors, var_name, ag_helpers[0].config.num_episodes)
            stats[ag_name][var_name] = ag_stats

    # prints comparison (csv, screen, bar-chart) between agents for the several variables
    print_mean_comparison(subdir, stats, var_names)
    print_evo_comparison(subdir, stats, var_names)

    # ============================================================
    # PART 2 : collects analysis stats for each agent
    subdir = join(output_dir, 'analysis')
    clear_dir(subdir)

    stats = OrderedDict()
    var_names = set()
    for ag_idx, analysis in enumerate(analyses):
        ag_name = ag_names[ag_idx]
        print_line('Collecting analysis stats for {}...'.format(ag_name), log_file)

        # collect stats for each analysis type
        stats[ag_name] = {}
        for group, analysis_stats in analysis.get_stats_grouped().items():
            # creates sub-dir for analysis group
            clear_dir(join(subdir, group))
            for var_name, stat in analysis_stats.items():
                var_name = join(group, var_name)
                var_names.add(var_name)
                stats[ag_name][var_name] = np.array([stat[0]]), np.array([stat[1]]), np.array([stat[2]])

    # prints comparison (csv, screen, bar-chart) between agents for the several variables
    print_mean_comparison(subdir, stats, var_names)

    log_file.close()
