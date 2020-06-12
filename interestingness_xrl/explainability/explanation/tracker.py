__author__ = 'Pedro Sequeira'

import numpy as np
import pygame
from os import makedirs
from os.path import join, exists
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from interestingness_xrl.explainability.explanation import Explainer, overlay_frame
from interestingness_xrl.learning import get_discretized_index

MIN_ALPHA = 150
HIGHLIGHT_COLOR = (215, 0, 0)


class AspectsTrackerExplainer(Explainer):
    """
    Corresponds to an explainer producing explanation videos, each visually tracking some interestingness element.
    """

    def __init__(self, env, helper, full_analysis, output_dir, recorded_episodes, fps):
        """
        Creates a new aspects tracker explainer.
        :param Env env: the Gym environment to be tracked, from which the frames are extracted.
        :param ScenarioHelper helper: the helper containing all necessary methods to extract information from the env.
        :param FullAnalysis full_analysis: the full analysis over the agent's history of interaction with the environment.
        :param str output_dir: the path to the output directory in which to save the videos.
        :param list recorded_episodes: the episodes in which episodes are to be recorded.
        :param int fps: the frames-per-second at which videos are to be recorded.
        """
        super().__init__(env, helper, full_analysis, output_dir, recorded_episodes)

        self.fps = fps

        # initializes structures
        pygame.init()
        self.video_recorders = {}
        self.maps = {}
        self.aspects_cache = [None for _ in range(self.config.num_states)]
        self.cur_time_step_frames = []

    def new_analysis_episode(self, e, length):
        super().new_analysis_episode(e, length)

        # finalize previous episode videos
        self.close()

        # ignores if not recording episode
        if e not in self.recorded_episodes:
            self.env.env.monitor = None
            return
        self.env.env.monitor = self.monitor

        # gets shape from rendered frame
        video_frame_size = self.env.render(mode='rgb_array').shape
        self.video_recorders = {}
        for group, aspect_names in self.full_analysis.get_interestingness_names_grouped().items():
            # creates output paths
            dir_path = join(self.output_dir, group)
            if not exists(dir_path):
                makedirs(dir_path)

            # creates video recorders and maps for each interestingness aspect
            for aspect_name in aspect_names:
                file_path = join(dir_path, '{}-e-{}.mp4'.format(aspect_name, e))
                self.video_recorders[aspect_name] = ImageEncoder(file_path, video_frame_size, self.fps, self.fps)
                self.maps[aspect_name] = np.zeros(self.config.env_size, np.int)

        self.cur_time_step_frames = []

    def _new_frame(self, frame):
        # adds frame to the current time-step list
        self.cur_time_step_frames.append(frame)

    def update_analysis(self, t, obs, s, a, r, ns):

        # ignores episodes not recorded
        if self.e not in self.recorded_episodes:
            return

        print('t: {}'.format(t))

        # captures an additional frame
        self.capture_frame()

        # gets maps for each aspect by placing agent in each possible location
        self._reset_maps()
        for col in range(self.num_cols):
            for row in range(self.num_rows):
                x, y = self.helper.get_cell_coordinates(col, row)

                # gets state from observation in desired location
                obs_vec = self.helper.get_features_from_observation(obs, x, y)
                s = get_discretized_index(obs_vec, self.feats_nbins)

                # for each interesting aspect found for this state (location)
                for aspect_name in self._get_aspects(s):
                    # increment counter for respective map
                    self.maps[self._get_video_name_key(aspect_name)][col][row] += 1

        # overlays maps for each aspect tracked for each frame
        for aspect_name in self.maps.keys():
            self._overlay_time_step_frames(aspect_name)

        # resets frame buffer
        self.cur_time_step_frames = []

    def _get_aspects(self, s):
        if self.aspects_cache[s] is None:
            # collects aspects found for the given state (location) and any possible action
            aspects = []
            for a in range(self.config.num_actions):
                for _, aspect_names in self.full_analysis.get_interesting_aspects_grouped(s, a, np.nan, -1).items():
                    aspects.extend(aspect_names)
            self.aspects_cache[s] = aspects
        return self.aspects_cache[s]

    def _overlay_time_step_frames(self, aspect_name):

        # creates the overlay surface with the colored squares from the map values
        overlay_surf = pygame.Surface(
            [self.num_cols * self.cell_width, self.num_rows * self.cell_height], pygame.SRCALPHA)
        for c in range(self.num_cols):
            x = c * self.cell_width
            for r in range(self.num_rows):
                y = r * self.cell_height
                alpha = self.maps[aspect_name][c][r].item()
                if alpha > 0:
                    alpha += MIN_ALPHA
                color = HIGHLIGHT_COLOR + (alpha,)
                overlay_surf.fill(color, pygame.Rect(x, y, x + self.cell_width, y + self.cell_height))

        # overlays map in each frame for this time-step
        for frame in self.cur_time_step_frames:
            frame = overlay_frame(frame, overlay_surf, (0, self.cell_height))
            self.video_recorders[aspect_name].capture_frame(frame)

    def _reset_maps(self):
        # zero all aspect maps
        for aspect_name in list(self.maps.keys()):
            self.maps[aspect_name] = np.zeros(self.config.env_size, np.int)

    def _get_video_name_key(self, aspect_name):
        try:
            return next(key for key in self.video_recorders.keys() if key in aspect_name)
        except StopIteration as e:
            print(e)  # should not get here

    def close(self):
        # close all videos
        for video_recorder in self.video_recorders.values():
            video_recorder.close()
        self.video_recorders = {}
