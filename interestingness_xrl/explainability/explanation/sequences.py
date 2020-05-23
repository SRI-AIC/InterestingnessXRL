__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
import pandas as pd
from os.path import join
from collections import deque
from gym import Env
from interestingness_xrl.util import record_video
from interestingness_xrl.explainability.explanation import Explainer, fade_frame

VIDEO_EXTENSION = 'mp4'
NUMPY_EXTENSION = 'npz'


class SequencesExplainer(Explainer):
    """
    Allows recording videos highlighting important action sequences by keeping track of an environment's frames.
    """

    def __init__(self, env, helper, full_analysis, output_dir, recorded_episodes,
                 fps, additional_time_steps, max_videos_per_sequence, max_between_states):
        """
        Creates a new highlights recorder.
        :param Env env: the Gym environment to be tracked, from which the frames are extracted.
        :param ScenarioHelper helper: the helper containing all necessary methods to extract information from the env.
        :param FullAnalysis full_analysis: the full analysis over the agent's history of interaction with the environment.
        :param str output_dir: the path to the output directory in which to save the videos.
        :param list recorded_episodes: the episodes in which episodes are to be recorded.
        :param int fps: the frames-per-second at which videos are to be recorded.
        :param int additional_time_steps: the number of environment time-steps to be recorded before and after a sequence in each video.
        :param int max_videos_per_sequence: the maximum number of highlights to be recorded for the same file name.
        :param int max_between_states: maximum number of not-in-sequence states allowed between 2 states in a sequence.
        """
        super().__init__(env, helper, full_analysis, output_dir, recorded_episodes)

        self.fps = fps
        self.additional_time_steps = additional_time_steps
        self.max_videos_per_sequence = max_videos_per_sequence
        self.max_between_sequence_states = max_between_states

        # initializes structures
        self.total_videos = {}
        self.captured_videos = set()

        # creates a dictionary containing the several possible sequences to be recorded
        self.sequences = {}
        for i, seq_info in enumerate(self.full_analysis.sequence_analysis.certain_seqs_to_subgoal):
            name, trans_seq = self.get_sequence_name(i, seq_info)
            self.sequences[name] = trans_seq

        # initializes structures
        self.candidate_highlights = {}
        self.cur_time_step_frames = []
        self.time_step_frames_buffer = []
        self.episodes_record_infos = []
        self.highlights_frames = {}

        self.active_sequences = {}
        self.active_sequences_idxs = {}
        self.active_sequences_between_states = {}

    def new_analysis_episode(self, e, length):
        super().new_analysis_episode(e, length)

        # turns-off frame collection for analysis
        self._set_collect_frames(False)

        self.active_sequences_idxs = {}
        self.active_sequences_between_states = {}
        self.cur_time_step_frames = []

    def update_analysis(self, t, obs, s, a, r, ns):

        # ignore new or active sequences if short length
        if t + self.additional_time_steps >= self.e_length:
            return

        # checks to see if old sequences have updated or have to be discarded
        for seq_name, idx in list(self.active_sequences_idxs.items()):

            # if end state of transition is the same, continue looking, sequence did not advance
            _, _, nst = self.sequences[seq_name][idx]
            if nst == ns:
                continue

            # if a state transition was achieved, increment idx
            advanced = False
            for i in range(len(self.sequences[seq_name]) - 1, idx - 1, -1):
                st, ac, nst = self.sequences[seq_name][i]
                # if st == s and ac == a and nst == ns:
                # if ac == a and nst == ns:
                if nst != ns:
                    continue

                self.active_sequences_idxs[seq_name] = i
                self.active_sequences_between_states[seq_name] = 0
                advanced = True

                # check sequence ended
                if i != len(self.sequences[seq_name]) - 1:
                    continue

                seq_obs, initial_time_step = self.active_sequences[seq_name]
                final_time_step = t + self.additional_time_steps
                seq_info = (seq_obs, initial_time_step, final_time_step, self.e)

                # checks max budget for this sequence and whether to replace some sequence with a better one
                highlight_idx = len(self.candidate_highlights[seq_name]) \
                    if seq_name in self.candidate_highlights else 0
                if highlight_idx == self.max_videos_per_sequence:

                    # gets distances between all candidate sequences
                    seq_distances = self._get_seq_distances(seq_name, seq_info)

                    # iteratively tries to improve diversity by replacing one of the highlights
                    highlight_idx = -1
                    max_diversity = self._get_diversity(seq_distances, self.max_videos_per_sequence)
                    for seq_idx in range(self.max_videos_per_sequence):
                        diversity = self._get_diversity(seq_distances, seq_idx)
                        if diversity > max_diversity:
                            max_diversity = diversity
                            highlight_idx = seq_idx

                # if there was improvement, adds highlight recording info to the list at the correct index
                if highlight_idx != -1:

                    if seq_name not in self.candidate_highlights:
                        self.candidate_highlights[seq_name] = [None]
                    elif highlight_idx == len(self.candidate_highlights[seq_name]):
                        self.candidate_highlights[seq_name].append(None)

                    self.candidate_highlights[seq_name][highlight_idx] = seq_info

                # stop tracking this sequence
                self._stop_tracking(seq_name)
                break

            # otherwise increment in-between states
            if not advanced:
                self.active_sequences_between_states[seq_name] += 1

                # if sequence is no longer valid, stop tracking it
                if self.active_sequences_between_states[seq_name] > self.max_between_sequence_states:
                    self._stop_tracking(seq_name)

        # tries to see if a new sequence has initiated
        for seq_name, seq in self.sequences.items():

            # ignore active sequences
            if seq_name in self.active_sequences_idxs:
                continue

            # ignore sequences with a maximum number of videos already recorded
            if seq_name in self.total_videos and self.total_videos[seq_name] == self.max_videos_per_sequence:
                continue

            # ignore sequences already recorded in this episode
            if seq_name in self.captured_videos:
                continue

            # if first transition was achieved, start tracking sequence
            st, ac, nst = self.sequences[seq_name][0]
            if st == s and ac == a and nst == ns:
                self._start_tracking(seq_name, t, obs)

    def finalize_analysis(self):

        sequences_infos = []

        # organizes recordings by episode and time-step
        self.episodes_record_infos = [[] for _ in range(self.e + 1)]
        for seq_name, record_infos in self.candidate_highlights.items():
            for idx, record_info in enumerate(record_infos):
                # retrieves highlight record info
                obs, initial_time_step, final_time_step, episode = record_info

                # adds record info to list
                self.episodes_record_infos[episode].append((initial_time_step, final_time_step, seq_name, idx))
                sequences_infos.append([seq_name, idx, episode, initial_time_step, final_time_step])

        # sorts and converts lists to queues
        for episode, record_infos in enumerate(self.episodes_record_infos):
            self.episodes_record_infos[episode].sort(key=lambda i: i[0])
            self.episodes_record_infos[episode] = deque(self.episodes_record_infos[episode])

        # save highlight frames to csv file in format (aspect, idx, episode, frame_start, frame_end)
        file_name = join(self.output_dir, 'highlights.csv')
        df = pd.DataFrame(sequences_infos,
                          columns=('sequence', 'idx', 'episode', 'initial time-step', 'final time-step'))
        df.to_csv(file_name, index=False)

    def new_explain_episode(self, e, length):
        super().new_explain_episode(e, length)

        self.time_step_frames_buffer = []
        self.cur_time_step_frames = []

        # activates frame recording only if episode contains highlight from first time-step
        if len(self.episodes_record_infos[self.e]) > 0:
            initial_time_step, *_ = self.episodes_record_infos[self.e][0]
            self._set_collect_frames(initial_time_step == 0)

    def update_explanation(self, t, obs, s, a, r, ns):

        # captures an additional frame
        if self._is_collecting_frames():
            self.capture_frame()

        # adds time-step frames to buffer
        self.time_step_frames_buffer.append(self.cur_time_step_frames)
        self.cur_time_step_frames = []

        # checks existing recording times
        while len(self.episodes_record_infos[self.e]) > 0:
            initial_time_step, final_time_step, seq_name, idx = self.episodes_record_infos[self.e][0]

            # if first record time hasn't been reached, no need to search further
            if final_time_step > t:
                break

            # pops info from queue
            self.episodes_record_infos[self.e].popleft()

            # adds frames to highlight list for this sequence
            if seq_name not in self.highlights_frames:
                self.highlights_frames[seq_name] = [[] for _ in range(len(self.candidate_highlights[seq_name]))]
            self.highlights_frames[seq_name][idx] = self.time_step_frames_buffer[initial_time_step:final_time_step + 1]

        # checks if frame recording needs to be activated
        self._set_collect_frames(False)
        if len(self.episodes_record_infos[self.e]) > 0:
            initial_time_step, *_ = self.episodes_record_infos[self.e][0]
            if t >= initial_time_step - 1:
                self._set_collect_frames(True)

    def close(self):

        # records all highlights to video files
        for seq_name in self.highlights_frames.keys():
            print('Creating {} videos...'.format(seq_name))

            # creates faded highlight and records individual highlight video
            for idx, time_steps_frames in enumerate(self.highlights_frames[seq_name]):
                file_name = join(self.output_dir, '{}-video{}'.format(seq_name, idx))
                highlight = self._create_highlight(time_steps_frames)
                self._record_video(file_name, highlight)

    def _new_frame(self, frame):
        # adds frame to the current time-step list
        self.cur_time_step_frames.append(frame)

    def _start_tracking(self, name, start_time, obs):
        self.active_sequences[name] = (obs, max(0, start_time - self.additional_time_steps))
        self.active_sequences_idxs[name] = 0
        self.active_sequences_between_states[name] = 0

    def _stop_tracking(self, name):
        del self.active_sequences[name]
        del self.active_sequences_idxs[name]
        del self.active_sequences_between_states[name]

    @staticmethod
    def get_sequence_name(seq_idx, seq_info):
        s, _, seq, _ = seq_info
        name = '#{}-s{}'.format(seq_idx, s)
        trans_seq = []
        for a, ns in seq:
            name += '-a{}-s{}'.format(a, ns)
            trans_seq.append((s, a, ns))
            s = ns
        return name, trans_seq

    @staticmethod
    def get_sequence_file_name(seq_name, video_num):
        return '{}-video{}.{}'.format(seq_name, video_num, VIDEO_EXTENSION)

    def _get_seq_distances(self, seq_name, new_seq_info):
        """
        Creates a table with the dissimilarities between all sequences of the given name.
        :param str seq_name: the name of the sequence.
        :param tuple new_seq_info: the information about the new candidate sequence in the form
        (seq_obs, initial_time_step, final_time_step, episode).
        :rtype: np.ndarray
        :return: a table with the dissimilarities between all sequences.
        """

        seq_infos = self.candidate_highlights[seq_name] + [new_seq_info]
        obs_distances = np.zeros((len(seq_infos), len(seq_infos)))
        for i in range(len(seq_infos)):
            len_i = seq_infos[i][2] - seq_infos[i][1]
            obs_i = seq_infos[i][0]
            for j in range(i + 1, len(seq_infos)):
                len_j = seq_infos[j][2] - seq_infos[j][1]
                obs_j = seq_infos[j][0]

                # first criterion is diff in sequence length
                dist = abs(len_i - len_j) / self.config.max_steps_per_episode

                # second is diff of observation of first step in sequence
                if dist == 0:
                    dist = self.helper.get_observation_dissimilarity(obs_i, obs_j)

                obs_distances[i][j] = dist
        return obs_distances

    @staticmethod
    def _get_diversity(distances, exclude_idx):
        """
        Calculates the diversity of the given observations.
        :param np.ndarray distances: a table with the dissimilarities between all sequences.
        :param int exclude_idx: the index of the sequence to be excluded from the diversity calculation.
        :rtype: float
        :return: a number in [0,1] indicating the diversity of the sequences, given by the product between the
        maximum and the minimum sequence distance.
        """
        min_dist = 2.
        max_dist = -1.
        for i in range(distances.shape[0]):
            if i == exclude_idx:
                continue
            for j in range(i + 1, distances.shape[0]):
                if j == exclude_idx:
                    continue
                dist = distances[i][j]
                min_dist = min(min_dist, dist)
                max_dist = max(max_dist, dist)
        return max_dist * min_dist

    def _record_video(self, file_name, frames, save_np_binary=True):
        """
        Saves the given frames into a video file and (optionally) a numpy binary compressed file.
        :param str file_name: the path to the file, without extension.
        :param list frames: a list containing the sequence of frames to be saved.
        :param bool save_np_binary: whether to save the frames into a numpy binary file.
        :return:
        """
        if save_np_binary:
            np.savez_compressed('{}.{}'.format(file_name, NUMPY_EXTENSION), frames)
        record_video(frames, '{}.{}'.format(file_name, VIDEO_EXTENSION), self.fps)

    def _create_highlight(self, time_steps_frames):
        """
        Creates a fade-in / fade-out frame sequence from the given frame array.
        :param list time_steps_frames: the list containing all time-steps frames.
        :rtype: list
        :return: a list of frames with fade-in / fade-out effects.
        """
        num_frames = np.array([len(time_step_frames) for time_step_frames in time_steps_frames])
        total_frames = np.sum(num_frames)
        fade_in_frames = np.sum(num_frames[:self.additional_time_steps])
        fade_out_frames = np.sum(num_frames[len(num_frames) - self.additional_time_steps:])

        i = 0
        frame_list = []
        for time_step_frames in time_steps_frames:
            for frame in time_step_frames:
                # checks if frame is simply to be copied to the video or faded
                if i < fade_in_frames:
                    frame = fade_frame(frame, 1 - (i / fade_in_frames))
                elif i >= total_frames - fade_out_frames:
                    frame = fade_frame(frame, (i - (total_frames - fade_out_frames)) / fade_out_frames)

                # adds frame to list
                frame_list.append(frame)
                i += 1

        return frame_list
