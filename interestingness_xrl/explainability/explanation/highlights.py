__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
import pandas as pd
from os import makedirs
from os.path import join, exists
from collections import deque
from gym import Env
from interestingness_xrl.util import record_video
from interestingness_xrl.explainability.explanation import Explainer, fade_frame

FADE_STEPS_PERCENT = .25  # percentage of total video time-steps used for fade-in/fade-out
VIDEO_EXTENSION = 'mp4'
NUMPY_EXTENSION = 'npz'


class HighlightsExplainer(Explainer):
    """
    Allows recording videos highlighting important times of the agent's interaction with its environment by keeping
    track of an environment's frames and of requests to record videos starting from specific time-steps.
    """

    def __init__(self, env, helper, full_analysis, output_dir, recorded_episodes,
                 fps, record_time_steps, max_highlights_per_aspect):
        """
        Creates a new highlights recorder.
        :param Env env: the Gym environment to be tracked, from which the frames are extracted.
        :param ScenarioHelper helper: the helper containing all necessary methods to extract information from the env.
        :param FullAnalysis full_analysis: the full analysis over the agent's history of interaction with the environment.
        :param str output_dir: the path to the output directory in which to save the videos.
        :param list recorded_episodes: the episodes in which episodes are to be recorded.
        :param float fps: the frames-per-second at which videos are to be recorded.
        :param int record_time_steps: the number of environment time-steps to be recorded in each video.
        :param int max_highlights_per_aspect: the maximum number of highlights to be recorded for the same file name.
        """
        super().__init__(env, helper, full_analysis, output_dir, recorded_episodes)

        self.record_time_steps = record_time_steps
        self.timer_time_steps = int((record_time_steps - 1) / 2)
        self.max_highlights_per_aspect = max_highlights_per_aspect
        self.fps = fps

        # initializes structures
        self.candidate_highlights = {}
        self.active_aspects = set()
        self.time_step_frames_buffer = deque(maxlen=record_time_steps)
        self.cur_time_step_frames = []
        self.episodes_record_infos = []
        self.highlights_frames = {}

        # creates output paths and group dictionary
        self.aspect_group = {}
        for group, aspect_names in self.full_analysis.get_interestingness_names_grouped().items():
            dir_path = join(self.output_dir, group)
            if not exists(dir_path):
                makedirs(dir_path)

            for aspect_name in aspect_names:
                self.aspect_group[aspect_name] = group

    def new_analysis_episode(self, e, length):
        super().new_analysis_episode(e, length)

        # turns-off frame collection for analysis
        self._set_collect_frames(False)

        self.active_aspects = set()

    def update_analysis(self, t, obs, s, a, r, ns):

        # checks interesting elements for current state and adds each to recording times queue
        for _, aspects in self.full_analysis.get_interesting_aspects_grouped(s, a, r, ns).items():
            for aspect_name in aspects:

                # gets final (record) time-step
                initial_time_step = t - self.timer_time_steps
                final_time_step = t + self.timer_time_steps

                # ignore if there is an active highlight being recorded for this aspect or if short length
                if aspect_name in self.active_aspects or \
                        final_time_step < self.record_time_steps or \
                        final_time_step >= self.e_length:
                    continue

                # checks max highlight budget for this aspect and whether to replace some highlight with a better one
                highlight_idx = len(self.candidate_highlights[aspect_name]) \
                    if aspect_name in self.candidate_highlights else 0
                if highlight_idx == self.max_highlights_per_aspect:

                    # gets distances between all candidate highlight observations
                    obs_list = [obs for obs, *_ in self.candidate_highlights[aspect_name]] + [obs]
                    obs_distances = self._get_obs_distances(obs_list)

                    # iteratively tries to improve diversity by replacing one of the highlights
                    highlight_idx = -1
                    max_diversity = self._get_obs_diversity(obs_distances, self.max_highlights_per_aspect)
                    for i in range(self.max_highlights_per_aspect):
                        diversity = self._get_obs_diversity(obs_distances, i)
                        if diversity > max_diversity:
                            max_diversity = diversity
                            highlight_idx = i

                    # if there was no improvement, ignore highlight of this observation
                    if highlight_idx == -1:
                        continue

                # adds highlight recording info to the list at the correct index
                if aspect_name not in self.candidate_highlights:
                    self.candidate_highlights[aspect_name] = [None]
                elif highlight_idx == len(self.candidate_highlights[aspect_name]):
                    self.candidate_highlights[aspect_name].append(None)

                self.candidate_highlights[aspect_name][highlight_idx] = \
                    (obs, initial_time_step, final_time_step, self.e)
                self.active_aspects.add(aspect_name)

        # checks whether aspects can be released if already finished recording in this episode
        for aspect_name, record_infos in self.candidate_highlights.items():
            if aspect_name in self.active_aspects:
                for idx, record_info in enumerate(record_infos):
                    _, _, final_time_step, episode = record_info
                    if episode == self.e and final_time_step <= t:
                        self.active_aspects.remove(aspect_name)
                        break

    def finalize_analysis(self):

        highlights_infos = []

        # organizes recordings by episode and time-step
        self.episodes_record_infos = [[] for _ in range(self.e + 1)]
        for aspect_name, record_infos in self.candidate_highlights.items():
            for idx, record_info in enumerate(record_infos):
                # retrieves highlight record info
                obs, initial_time_step, final_time_step, episode = record_info

                # adds record info to list and data-frame
                self.episodes_record_infos[episode].append((initial_time_step, final_time_step, aspect_name, idx))
                highlights_infos.append([aspect_name, idx, episode, initial_time_step, final_time_step])

        # sorts and converts lists to queues
        for episode, record_infos in enumerate(self.episodes_record_infos):
            self.episodes_record_infos[episode].sort(key=lambda i: i[0])
            self.episodes_record_infos[episode] = deque(self.episodes_record_infos[episode])

        # save highlight frames to csv file in format (aspect, idx, episode, frame_start, frame_end)
        file_name = join(self.output_dir, 'highlights.csv')
        df = pd.DataFrame(highlights_infos,
                          columns=('aspect', 'idx', 'episode', 'initial time-step', 'final time-step'))
        df.to_csv(file_name, index=False)

    def new_explain_episode(self, e, length):
        super().new_explain_episode(e, length)

        self.time_step_frames_buffer = deque(maxlen=self.record_time_steps)
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
        time_steps_frames = list(self.time_step_frames_buffer)

        # checks existing recording times
        while len(self.episodes_record_infos[self.e]) > 0:
            initial_time_step, final_time_step, aspect_name, idx = self.episodes_record_infos[self.e][0]

            # if first record time hasn't been reached, no need to search further
            if final_time_step > t:
                break

            # pops info from queue
            self.episodes_record_infos[self.e].popleft()

            # adds frames to highlight list for this aspect
            if aspect_name not in self.highlights_frames:
                self.highlights_frames[aspect_name] = [[] for _ in range(len(self.candidate_highlights[aspect_name]))]
            self.highlights_frames[aspect_name][idx] = time_steps_frames

        # checks if frame recording needs to be activated
        self._set_collect_frames(False)
        if len(self.episodes_record_infos[self.e]) > 0:
            initial_time_step, *_ = self.episodes_record_infos[self.e][0]
            if t >= initial_time_step - 1:
                self._set_collect_frames(True)

    def close(self):

        # records all highlights to video files
        for aspect_name in self.highlights_frames.keys():
            print('Creating {} videos...'.format(aspect_name))

            group = self._get_aspect_group(aspect_name)
            highlights = []
            for idx, time_steps_frames in enumerate(self.highlights_frames[aspect_name]):
                # creates faded highlight and records individual highlight video
                file_name = join(self.output_dir, group, '{}-video{}'.format(aspect_name, idx))
                highlight = self._create_highlight(time_steps_frames)
                self._record_video(file_name, highlight)

                # adds highlight do list
                highlights.extend(highlight)

            # creates video with sequence of all highlights for aspect
            file_name = join(self.output_dir, group, '{}-compact'.format(aspect_name))
            self._record_video(file_name, highlights, False)

    def _new_frame(self, frame):
        # adds frame to the current time-step list
        self.cur_time_step_frames.append(frame)

    def _get_aspect_group(self, aspect_name):
        """
        Gets the name of the group to which thw given aspect belongs.
        :param str aspect_name: the name of the interesting element.
        :rtype: str
        :return: the name of the group to which thw given aspect belongs.
        """
        try:
            aspect_key = next(key for key in self.aspect_group.keys() if key in aspect_name)
            return self.aspect_group[aspect_key]
        except StopIteration as e:
            print(e)  # should not get here

    def _get_obs_distances(self, obs_list):
        """
        Creates a table with the dissimilarities between all given observations.
        :param list obs_list: a list with np.ndarray observations.
        :rtype: np.ndarray
        :return: a table with the dissimilarities between all given observations.
        """
        obs_distances = np.zeros((len(obs_list), len(obs_list)))
        for i in range(len(obs_list)):
            for j in range(i + 1, len(obs_list)):
                obs_distances[i][j] = self.helper.get_observation_dissimilarity(obs_list[i], obs_list[j])
        return obs_distances

    @staticmethod
    def _get_obs_diversity(obs_distances, exclude_idx):
        """
        Calculates the diversity of the given observations.
        :param np.ndarray obs_distances: a table with the dissimilarities between all given observations.
        :param int exclude_idx: the index of the observation to be excluded from the diversity calculation.
        :rtype: float
        :return: a number in [0,1] indicating the diversity of the observations, given by the product between the
        maximum and the minimum observation distance.
        """
        min_dist = 2.
        max_dist = -1.
        for i in range(obs_distances.shape[0]):
            if i == exclude_idx:
                continue
            for j in range(i + 1, obs_distances.shape[0]):
                if j == exclude_idx:
                    continue
                dist = obs_distances[i][j]
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
        fade_steps = int(FADE_STEPS_PERCENT * self.record_time_steps)
        fade_in_frames = np.sum(num_frames[:fade_steps])
        fade_out_frames = np.sum(num_frames[len(num_frames) - fade_steps:])

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
