__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import sys
from os import makedirs
from os.path import join, exists, split
from interestingness_xrl.explainability.analysis import AnalysisBase
from interestingness_xrl.scenarios import FroggerConfiguration

ENV_PREFIX = 'frogger'
OUTPUT_DIR = './results/explanations/' + ENV_PREFIX
FULL_ANALYSIS_FILE = 'full-analysis.json'
AGENT_1_RESULTS_DIR = './results/explanations/' + ENV_PREFIX + '/train'
AGENT_2_RESULTS_DIR = './results/explanations/' + ENV_PREFIX + '/test'

if __name__ == '__main__':

    # tries to load full analyses from results dirs
    results_dir1 = sys.argv[1] if len(sys.argv) > 1 else AGENT_1_RESULTS_DIR
    file_name = join(results_dir1, FULL_ANALYSIS_FILE)
    if not exists(file_name):
        raise ValueError('Full analysis not found: {}'.format(file_name))
    full_analysis1 = AnalysisBase.load_json(file_name)

    config_file = join(results_dir1, 'config.json')
    if not exists(config_file):
        raise ValueError('Configuration not found: {}'.format(config_file))
    full_analysis1.config = FroggerConfiguration.load_json(config_file)

    results_dir2 = sys.argv[2] if len(sys.argv) > 2 else AGENT_2_RESULTS_DIR
    file_name = join(results_dir2, FULL_ANALYSIS_FILE)
    if not exists(file_name):
        raise ValueError('Full analysis not found: {}'.format(file_name))
    full_analysis2 = AnalysisBase.load_json(file_name)

    config_file = join(results_dir2, 'config.json')
    if not exists(config_file):
        raise ValueError('Configuration not found: {}'.format(config_file))
    full_analysis2.config = FroggerConfiguration.load_json(config_file)

    # creates output dir if needed
    output_dir = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_DIR
    if not exists(output_dir):
        makedirs(output_dir)

    # saves / copies configs to file
    full_analysis1.config.save_json(join(output_dir, 'config1.json'))
    full_analysis2.config.save_json(join(output_dir, 'config2.json'))

    # gets difference between 2 and 1 (added elements)
    analysis_diff = full_analysis2.difference_to(full_analysis1)
    analysis_diff.save_json(join(output_dir, '{}-{}.json'.format(split(results_dir2)[-1], split(results_dir1)[-1])))
    analysis_diff.save_report(
        join(output_dir, '{}-{}.txt'.format(split(results_dir2)[-1], split(results_dir1)[-1])), True)

    # gets difference between 1 and 2 (removed elements)
    analysis_diff = full_analysis1.difference_to(full_analysis2)
    analysis_diff.save_json(join(output_dir, '{}-{}.json'.format(split(results_dir1)[-1], split(results_dir2)[-1])))
    analysis_diff.save_report(
        join(output_dir, '{}-{}.txt'.format(split(results_dir1)[-1], split(results_dir2)[-1])), True)
