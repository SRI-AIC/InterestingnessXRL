__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import pygame
import numpy as np
from os import makedirs
from os.path import join, exists
from palettable.cmocean.sequential import get_map
from interestingness_xrl.explainability.explanation import Explainer

AGENT_LOCATION_HEATMAP = 'agent location'

FEATURES_SUB_DIR = 'features'
STATS_SUB_DIR = 'stats'
ANALYSIS_SUB_DIR = 'analysis'

# COLOR_MAP = get_map('BluGrn_7').mpl_colormap
COLOR_MAP = get_map('Haline_20').mpl_colormap


class HeatmapsExplainer(Explainer):
    """
    Corresponds to an explainer producing heatmap visual explanations based on agent location frequencies, observation
    feature frequencies, game stats means and analysis elements frequencies.
    """

    def __init__(self, env, helper, full_analysis, output_dir, recorded_episodes):
        """
        Creates a new heatmaps explainer.
        :param Env env: the Gym environment to be tracked, from which the frames are extracted.
        :param ScenarioHelper helper: the helper containing all necessary methods to extract information from the env.
        :param FullAnalysis full_analysis: the full analysis over the agent's history of interaction with the environment.
        :param str output_dir: the path to the output directory in which to save the videos.
        :param list recorded_episodes: the episodes in which episodes are to be recorded.
        """
        super().__init__(env, helper, full_analysis, output_dir, recorded_episodes)

        # creates heatmaps sub-dirs
        makedirs(join(output_dir, FEATURES_SUB_DIR))
        makedirs(join(output_dir, STATS_SUB_DIR))
        makedirs(join(output_dir, ANALYSIS_SUB_DIR))

        # initializes heatmaps
        self.heat_maps = {AGENT_LOCATION_HEATMAP: np.zeros(self.config.env_size, np.int)}

        # creates mean heatmaps for each stat collected by the helper
        for var_name in helper.stats_collector.all_variables():
            name = self._get_heatmap_name(STATS_SUB_DIR, var_name)
            self.heat_maps[name] = np.zeros(self.config.env_size, np.float)

        # creates count heatmaps for each feature label location
        feats_nbins = helper.get_features_bins()
        for f in range(len(feats_nbins)):
            for v in range(feats_nbins[f]):
                name = self._get_heatmap_name(FEATURES_SUB_DIR, helper.get_feature_label(f, v))
                self.heat_maps[name] = np.zeros(self.config.env_size, np.int)

        # creates heatmaps for each analysis aspect / interestingness element
        self.aspect_name_keys = []
        for group_name, aspect_names in self.full_analysis.get_interestingness_names_grouped().items():
            dir_name = join(self.output_dir, ANALYSIS_SUB_DIR, group_name)
            if not exists(dir_name):
                makedirs(dir_name)
            for aspect_name in aspect_names:
                name = self._get_heatmap_name(join(ANALYSIS_SUB_DIR, group_name), aspect_name)
                self.aspect_name_keys.append(name)
                self.heat_maps[name] = np.zeros(self.config.env_size, np.int)

    def new_analysis_episode(self, e, length):
        # we don't need a monitor for heatmaps
        self.env.env.monitor = None

    def update_analysis(self, t, obs, s, a, r, ns):

        col, row = self.helper.get_agent_cell_location(obs)
        obs_vec = self.helper.get_features_from_observation(obs)

        # updates heatmaps
        prev_count = self.heat_maps[AGENT_LOCATION_HEATMAP][col][row]

        # updates stats heatmaps
        for var_name in self.helper.stats_collector.all_variables():
            hp_name = self._get_heatmap_name(STATS_SUB_DIR, var_name)
            prev_stat = self.heat_maps[hp_name][col][row]
            cur_stat = self.helper.stats_collector.get_most_recent_sample(var_name, self.e)
            self.heat_maps[hp_name][col][row] = self._get_mean(cur_stat, prev_stat, prev_count)

        self.heat_maps[AGENT_LOCATION_HEATMAP][col][row] += 1

        # updates feature heatmaps
        for f in range(len(self.feats_nbins)):
            feat_label = self.helper.get_feature_label(f, obs_vec[f])
            hp_name = self._get_heatmap_name(FEATURES_SUB_DIR, feat_label)
            self.heat_maps[hp_name][col][row] += 1

        # updates analyses heatmaps
        for group_name, aspect_names in self.full_analysis.get_interesting_aspects_grouped(s, a, r, ns).items():
            for aspect_name in aspect_names:
                aspect_name = \
                    self._get_heatmap_name_key(self._get_heatmap_name(join(ANALYSIS_SUB_DIR, group_name), aspect_name))
                self.heat_maps[aspect_name][col][row] += 1

    def close(self):
        # normalize heat-maps and saves to file
        for name, heat_map in self.heat_maps.items():
            self._save_heat_map(self._normalize(heat_map), join(self.output_dir, '{}.png'.format(name.lower())))

    @staticmethod
    def _get_heatmap_name(group_name, aspect_name):
        return join(group_name, aspect_name)

    def _get_heatmap_name_key(self, aspect_name):
        try:
            return next(key for key in self.aspect_name_keys if key in aspect_name)
        except StopIteration as e:
            print(e)  # should not get here

    @staticmethod
    def _get_mean(sample, old_mean, n):
        return (sample + old_mean * n) / (n + 1)

    @staticmethod
    def _normalize(_map):
        map_range = np.max(_map) - np.min(_map)
        return _map if map_range == 0 else np.true_divide(_map - np.min(_map), map_range)

    def _save_heat_map(self, _map, file_path):

        # creates surface on which to draw
        pygame.init()
        surf = pygame.Surface([self.num_cols * self.cell_width, self.num_rows * self.cell_height], pygame.SRCALPHA, 32)

        # builds heatmap from the normalized values
        for c in range(self.num_cols):
            x = c * self.cell_width

            for r in range(self.num_rows):
                y = r * self.cell_height
                color = COLOR_MAP(_map[c][r], bytes=True)
                pygame.draw.rect(surf, color, pygame.Rect(x, y, x + self.cell_width, y + self.cell_height))

        # saves screenshot with heatmap
        pygame.image.save(surf, file_path)
