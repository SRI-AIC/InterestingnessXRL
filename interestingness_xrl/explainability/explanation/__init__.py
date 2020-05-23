__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
import pygame


class Explainer(object):
    """
    Defines a base class for methods of explanation, i.e., that provide or select xplanations based on agent updates
    while interacting with an environment.
    """

    def __init__(self, env, helper, full_analysis, output_dir, recorded_episodes):
        """
        Creates a new explainer.
        :param Env env: the Gym environment to be tracked, from which the frames are extracted.
        :param ScenarioHelper helper: the helper containing all necessary methods to extract information from the env.
        :param FullAnalysis full_analysis: the full analysis over the agent's history of interaction with the environment.
        :param str output_dir: the path to the output directory in which to save the videos.
        :param list recorded_episodes: the episodes in which episodes are to be recorded.
        """
        self.env = env
        self.helper = helper
        self.full_analysis = full_analysis
        self.output_dir = output_dir
        self.recorded_episodes = set(recorded_episodes)

        # useful info
        self.config = helper.config
        self.num_cols = self.config.env_size[0]
        self.num_rows = self.config.env_size[1]
        self.cell_width = self.config.cell_size[0]
        self.cell_height = self.config.cell_size[1]
        self.feats_nbins = helper.get_features_bins()

        self.e = 0
        self.e_length = 0
        self.monitor = None

    def capture_frame(self):
        """
        Captures a frame from the environment. Some explainers might want to do something with the new frame.
        This gets called by the Gym environment.
        :rtype: np.ndarray
        :return: the new image frame in a 3D pixel format.
        """
        frame = self.env.render(mode='rgb_array')
        self._new_frame(frame)

    def _set_collect_frames(self, collect):
        """
        Activates / deactivates frame collection from the Gym environment.
        :param bool collect: whether to collect frames.
        :return:
        """
        self.env.env.monitor = self.monitor if collect else None

    def _is_collecting_frames(self):
        """
        Checks whether frame collection is currently enabled.
        :rtype: bool
        :return: whether frame collection is currently enabled.
        """
        return self.env.env.monitor is not None

    def _new_frame(self, frame):
        """
        Signals that a new image frame was captured from the environment.
        :param np.ndarray frame: the array containing the image information.
        :return:
        """
        pass

    def new_analysis_episode(self, e, length):
        """
        Signals the start of a new episode in the analysis phase.
        :param int e: the index of the new episode.
        :param int length: the length of the new episode in time-steps.
        :return:
        """
        self.e = e
        self.e_length = length

    def update_analysis(self, t, obs, s, a, r, ns):
        """
        Updates the explainer according to the observed agent sample in the analysis phase.
        :param int t: the time-step at which the sample was observed.
        :param np.ndarray obs: the current observation provided to the agent.
        :param int s: the index of the previous state.
        :param int a: the index of the action executed by the agent.
        :param float r: the reward received by the agent.
        :param int ns: the index of the state to which the environment transitioned to.
        :return:
        """
        pass

    def new_explain_episode(self, e, length):
        """
        Signals the start of a new episode in the explanation phase.
        :param int e: the index of the new episode.
        :param int length: the length of the new episode in time-steps.
        :return:
        """
        self.e = e
        self.e_length = length

    def update_explanation(self, t, obs, s, a, r, ns):
        """
        Updates the explainer according to the observed agent sample in the explanation phase.
        :param int t: the time-step at which the sample was observed.
        :param np.ndarray obs: the current observation provided to the agent.
        :param int s: the index of the previous state.
        :param int a: the index of the action executed by the agent.
        :param float r: the reward received by the agent.
        :param int ns: the index of the state to which the environment transitioned to.
        :return:
        """
        pass

    def finalize_analysis(self):
        """
        Finalizes analysis phase and prepares for explanation phase.
        :return:
        """

    def close(self):
        """
        Terminates necessary operation of the explainer.
        :return:
        """
        pass


def overlay_frame(frame, overlay_surf, location):
    """
    Overlays a given surface (with transparency) into the given frame.
    :param np.ndarray frame: the original image frame.
    :param pygame.Surface overlay_surf: the overlay surface.
    :param tuple location: the (x, y) location in which to draw the overlay image over the original frame.
    :rtype: np.ndarray
    :return: an image frame with the overlayed surface.
    """
    # converts frame to pygame surface
    frame = np.rot90(np.fliplr(frame), 1)
    surf = pygame.surfarray.make_surface(frame)

    # merges the two surfaces
    surf.blit(overlay_surf, location)

    # converts surface back to 3d frame and adds to video recorder
    frame = pygame.surfarray.array3d(surf).astype(np.uint8)
    frame = np.fliplr(np.rot90(frame, 3))

    return frame


def fade_frame(frame, fade_level):
    """
    Fades the given frame by overlaying a black surface with transparency.
    :param np.ndarray frame: the original image frame.
    :param float fade_level: the fade level in [0,1], where 0 is completely transparent, 1 is an opaque black surface.
    :rtype: np.ndarray
    :return: the faded frame.
    """
    height, width, _ = frame.shape
    fade_surf = pygame.Surface([height, width], pygame.SRCALPHA)
    fade_surf.fill((0, 0, 0, int(fade_level * 255)))

    frame_surf = pygame.surfarray.make_surface(frame)
    frame_surf.blit(fade_surf, (0, 0))
    frame = pygame.surfarray.array3d(frame_surf).astype(np.uint8)

    return frame
