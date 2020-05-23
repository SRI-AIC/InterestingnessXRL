__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import sys
import shutil
import numpy as np
from os import makedirs
from os.path import join, exists, pardir
from shutil import rmtree
from interestingness_xrl.util import record_video, print_line
from interestingness_xrl.explainability.analysis.full_analysis import FullAnalysis
from interestingness_xrl.explainability.explanation.sequences import SequencesExplainer
from interestingness_xrl.explainability.explanation.highlights import NUMPY_EXTENSION, VIDEO_EXTENSION
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration
from interestingness_xrl.scenarios import get_agent_output_dir, create_helper, AgentType, get_analysis_output_dir, \
    DATA_ORGANIZER_CONFIGS, get_explanation_output_dir, ReportType, HIGHLIGHTS_FPS

# DEF_AGENT_TYPE = AgentType.Learning
DEF_AGENT_TYPE = AgentType.Testing

CLEAN_DIR = True
MAX_HIGHLIGHTS = 4  # maximum highlights per aspect type/combo (has to be pair)


def _collect_aspect_highlights(_aspect_name, aspect_lst, num_highlights):
    global aspect_group, highlights_dirs, ag_idx

    highlights = []
    aspect_idx = 0
    total_highlights = 0
    highlight_idx = 0

    # until highlight budget is not reached
    group_name = aspect_group[_aspect_name]
    while total_highlights < num_highlights and aspect_idx < len(aspect_lst):

        # tries to load highlight for the aspect
        s, *_ = aspect_lst[aspect_idx]
        highlight_file = join(highlights_dirs[ag_idx], group_name, '{}-{}-video{}.{}'.format(
            _aspect_name, s, highlight_idx, NUMPY_EXTENSION))
        if exists(highlight_file):
            highlights.append(np.load(highlight_file))
            total_highlights += 1
            highlight_idx += 1
            print_line('Loaded highlight from: {}'.format(highlight_file), log_file)
        else:
            # advances aspect element
            aspect_idx += 1
            highlight_idx = 0
    return highlights


def _save_highlights_video(highlights, sub_dir_name, _file_name):
    global log_file, output_dir

    sub_dir = join(output_dir, sub_dir_name)
    if not exists(sub_dir):
        makedirs(sub_dir)

    # tries to save highlight compact to video file
    if len(highlights) > 0:
        highlights_frames = []
        for frames in highlights:
            highlights_frames.extend(frames[frames.files[0]])
        video_file_path = join(sub_dir, '{}.{}'.format(_file_name.lower(), VIDEO_EXTENSION))
        record_video(highlights_frames, video_file_path, HIGHLIGHTS_FPS)
        print_line('Saved highlights compact in: {}'.format(video_file_path), log_file)


def _save_highlights_for_aspect(sub_dir, _aspect_name, aspect_lst, num_highlights, agent_name):
    global log_file

    print_line('==============================================', log_file)
    print_line('Collecting and saving highlights for aspect {} of \'{}\' in sub-dir: \'{}\''.format(
        _aspect_name, agent_name, sub_dir), log_file)

    # collect and saves highlights for this aspect
    highlights = _collect_aspect_highlights(_aspect_name, aspect_lst, num_highlights)
    _save_highlights_video(highlights, sub_dir, agent_name)


def _save_highlights_for_aspects(sub_dir, _aspect_names, aspect_lsts, num_highlights, agent_name):
    global log_file

    print_line('==============================================', log_file)
    print_line('Collecting and saving highlights for {} aspects of \'{}\' in sub-dir: \'{}\'...'.format(
        len(_aspect_names), agent_name, sub_dir), log_file)

    # collect and saves highlights for the different aspects
    highlights = []
    for _i in range(len(_aspect_names)):
        highlights.extend(_collect_aspect_highlights(_aspect_names[_i], aspect_lsts[_i], num_highlights[_i]))
    _save_highlights_video(highlights, sub_dir, agent_name)


def _save_sequence(sub_dir_name, sequence_lst, agent_name):
    global output_dir, log_file, ag_idx

    sub_dir = join(output_dir, sub_dir_name)
    if not exists(sub_dir):
        makedirs(sub_dir)

    print_line('==============================================', log_file)
    print_line('Collecting and saving sequence for \'{}\' in sub-dir: \'{}\''.format(agent_name, sub_dir), log_file)

    # tries to get the longest recorded sequences
    for seq_idx, seq_info in enumerate(sequence_lst):
        seq_name, _ = SequencesExplainer.get_sequence_name(seq_idx, seq_info)
        seq_file_name = join(sequences_dirs[ag_idx], SequencesExplainer.get_sequence_file_name(seq_name, 0))

        # if file exists, copy to results dir
        if exists(seq_file_name):
            video_file_path = join(sub_dir, '{}.{}'.format(agent_name.lower(), VIDEO_EXTENSION))
            shutil.copyfile(seq_file_name, video_file_path)
            print_line('Saved sequence by copying from: {}\n\tto: {}'.format(seq_file_name, video_file_path), log_file)
            return


if __name__ == '__main__':

    aspect_group = {}

    # tries to get agent type
    agent_t = int(sys.argv[1]) if len(sys.argv) > 1 else DEF_AGENT_TYPE

    # tries to load each analysis from results dir
    analyses = []
    highlights_dirs = []
    sequences_dirs = []
    ag_names = []
    for ag_name, config in DATA_ORGANIZER_CONFIGS.items():

        agent_dir = join(pardir, get_agent_output_dir(config, agent_t))
        config_file = join(agent_dir, 'config.json')
        if not exists(config_file):
            raise ValueError('Configuration not found: {}'.format(config_file))
        config = EnvironmentConfiguration.load_json(config_file)

        # creates env helper
        helper = create_helper(config)

        # tries to load full analysis
        analyses_dir = get_analysis_output_dir(agent_dir)
        file_name = join(analyses_dir, 'full-analysis.json')
        if exists(file_name):
            analysis = FullAnalysis.load_json(file_name)
        else:
            raise ValueError('Full analysis not found at: {}'.format(file_name))
        analysis.set_helper(helper)

        # checks highlights and sequences dirs
        highlights_dir = get_explanation_output_dir(agent_dir, ReportType.Highlights)
        if exists(highlights_dir):
            highlights_dirs.append(highlights_dir)
        else:
            raise ValueError('Highlights not found at: {}'.format(highlights_dir))

        sequences_dir = get_explanation_output_dir(agent_dir, ReportType.Sequences)
        if exists(sequences_dir):
            sequences_dirs.append(sequences_dir)
        else:
            raise ValueError('Sequences not found at: {}'.format(sequences_dir))

        # collects aspects and group names
        for group, aspect_names in analysis.get_interestingness_names_grouped().items():
            for aspect_name in aspect_names:
                aspect_group[aspect_name] = group

        print('Loaded full analysis and highlights for agent {}.'.format(config.name))
        analyses.append(analysis)
        ag_names.append(ag_name)

    # creates output dir if needed
    output_dir = sys.argv[2] if len(sys.argv) > 2 else join(pardir, 'results', 'highlights-data')
    if not exists(output_dir):
        makedirs(output_dir)
    elif CLEAN_DIR:
        rmtree(output_dir)
        makedirs(output_dir)

    # creates log file
    log_file = open(join(output_dir, 'log.txt'), 'w')

    # organizes highlight data by collecting highlights for each agent (type)
    for ag_idx, analysis in enumerate(analyses):
        # saves / copies configs to file
        ag_name = ag_names[ag_idx]
        analysis.config.save_json(join(output_dir, 'config-{}.json'.format(ag_name)))

        # for single aspects: collect 4 highlights of the best elements
        _save_highlights_for_aspect('0-local-maxima', 'local-maximum-s',
                                    analysis.trans_value_analysis.local_maxima_states, MAX_HIGHLIGHTS, ag_name)
        _save_highlights_for_aspect('1-local-minima', 'local-minimum-s',
                                    analysis.trans_value_analysis.local_minima_states, MAX_HIGHLIGHTS, ag_name)
        _save_highlights_for_aspect('2-certain-states', 'certain-exec-s',
                                    analysis.state_action_freq_analysis.certain_states, MAX_HIGHLIGHTS, ag_name)
        _save_highlights_for_aspect('3-uncertain-states', 'uncertain-exec-s',
                                    analysis.state_action_freq_analysis.uncertain_states, MAX_HIGHLIGHTS, ag_name)
        _save_highlights_for_aspect('4-frequent-states', 'frequent-s',
                                    analysis.state_freq_analysis.freq_states, MAX_HIGHLIGHTS, ag_name)
        _save_highlights_for_aspect('5-infrequent-states', 'infrequent-s',
                                    analysis.state_freq_analysis.infreq_states, MAX_HIGHLIGHTS, ag_name)

        # for dimensions: collect 2 highlights of each aspect
        _save_highlights_for_aspects('6-transition-values', ['local-maximum-s', 'local-minimum-s'],
                                     [analysis.trans_value_analysis.local_maxima_states,
                                      analysis.trans_value_analysis.local_minima_states],
                                     [int(MAX_HIGHLIGHTS / 2), int(MAX_HIGHLIGHTS / 2)], ag_name)
        _save_highlights_for_aspects('7-execution-certainty', ['certain-exec-s', 'uncertain-exec-s'],
                                     [analysis.state_action_freq_analysis.certain_states,
                                      analysis.state_action_freq_analysis.uncertain_states],
                                     [int(MAX_HIGHLIGHTS / 2), int(MAX_HIGHLIGHTS / 2)], ag_name)
        _save_highlights_for_aspects('8-state-frequency', ['frequent-s', 'infrequent-s'],
                                     [analysis.state_freq_analysis.freq_states,
                                      analysis.state_freq_analysis.infreq_states],
                                     [int(MAX_HIGHLIGHTS / 2), int(MAX_HIGHLIGHTS / 2)], ag_name)

        # mix with a highlight from each aspect
        _save_highlights_for_aspects(
            '9-all-aspects',
            ['local-maximum-s', 'local-minimum-s', 'certain-exec-s', 'uncertain-exec-s', 'frequent-s', 'infrequent-s'],
            [analysis.trans_value_analysis.local_maxima_states,
             analysis.trans_value_analysis.local_minima_states,
             analysis.state_action_freq_analysis.certain_states,
             analysis.state_action_freq_analysis.uncertain_states,
             analysis.state_freq_analysis.freq_states,
             analysis.state_freq_analysis.infreq_states],
            [1, 1, 1, 1, 1, 1], ag_name)

        # save longest sequence
        _save_sequence('10-sequences', analysis.sequence_analysis.certain_seqs_to_subgoal, ag_name)

    log_file.close()
