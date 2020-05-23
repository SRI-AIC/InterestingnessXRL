__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import pygame
import pandas as pd
from itertools import product
from os.path import join, dirname, abspath
from interestingness_xrl.learning import get_discretized_index
from interestingness_xrl.learning.stats_collector import StatType
from interestingness_xrl.scenarios.scenario_helper import ScenarioHelper, TIME_STEPS_VAR, ANY_FEATURE_IDX
from interestingness_xrl.scenarios.frogger.configurations import FroggerConfiguration
from interestingness_xrl.util import print_line, clean_console
from frogger import FroggerState, TICK_RWD_ATTR, HIT_WATER_RWD_ATTR, HIT_CAR_RWD_ATTR, CELL_WIDTH, CELL_HEIGHT, \
    ARRIVAL_POSITIONS, ARRIVAL_WIDTH, NO_LIVES_RWD_ATTR, NEW_LEVEL_RWD_ATTR, FROG_ARRIVED_RWD_ATTR, \
    ACTION_RIGHT_KEY, ACTION_LEFT_KEY, MIN_Y_POS, MAX_Y_POS, MAX_GRASS_Y_POS, NUM_COLS, NUM_ROWS, NOT_DEAD_IDX, \
    CAR_DEATH_IDX, WATER_DEATH_IDX, TIME_UP_DEATH_IDX
from frogger.gym import *

CSV_DELIMITER = ','

MAX_LEVEL = 6
MAX_POINTS = 150000

DIRS = ['left', 'right', 'up', 'down']
SHORT_DIRS = ['L', 'R', 'U', 'D']
LEFT_IDX = 0
RIGHT_IDX = 1
UP_IDX = 2
DOWN_IDX = 3

EMPTY_IDX = 0
WATER_IDX = 1
CAR_IDX = 2
LOG_IDX = 3
LILYPAD_IDX = 4
OUT_OF_BOUNDS_IDX = 5

WIN_OBS_VEC = [LILYPAD_IDX] * len(DIRS)
DEATH_OBS_VEC = [OUT_OF_BOUNDS_IDX] * len(DIRS)

ELEM_LABELS = {EMPTY_IDX: 'empty',
               WATER_IDX: 'water',
               CAR_IDX: 'car',
               LOG_IDX: 'log',
               LILYPAD_IDX: 'lilypad',
               OUT_OF_BOUNDS_IDX: 'out-bounds',
               ANY_FEATURE_IDX: 'any'}

GAME_POINTS_VAR = 'Game points'
GAME_LEVEL_VAR = 'Game level'
NUM_LIVES_VAR = 'Num lives'
NUM_DEATHS_VAR = 'Num deaths'
FROGS_ARRIVED_VAR = 'Frogs arrived at '
NUM_MOVES_LIFE_VAR = 'Num moves per life'

DEATH_ON_WATER_VAR = "Num deaths water"
DEATH_BY_CAR_VAR = "Num deaths car"
DEATH_MAX_MOVES_VAR = "Num deaths max moves"

DEATH_MIDDLE_GRASS_VAR = "Num deaths middle grass"
DEATH_BOTTOM_GRASS_VAR = "Num deaths bottom grass"
DEATH_ROAD_VAR = "Num deaths road"
DEATH_RIVER_VAR = "Num deaths river"

POS_MIDDLE_GRASS_VAR = "Position middle grass"
POS_BOTTOM_GRASS_VAR = "Position bottom grass"
POS_ROAD_VAR = "Position road"
POS_RIVER_VAR = "Position river"

# the variables that we print to the screen every episode
PRINT_SCREEN_VAR_NAMES = [TIME_STEPS_VAR, GAME_POINTS_VAR, NUM_LIVES_VAR, GAME_LEVEL_VAR]


class FroggerHelper(ScenarioHelper):
    """
    Represents a set of helper methods for learning and analysis of Frogger game environments in the egocentric /
    adjacent view agent observation scheme.
    """

    def __init__(self, config, img_format='png', sound=False):
        """
        Creates a new Frogger environment helper according to the given configuration.
        :param FroggerConfiguration config: the configuration with all necessary parametrization.
        :param str img_format: the format of the observations' image files.
        :param bool sound: whether to play game sounds.
        """
        super().__init__(config)

        self.img_format = img_format
        self.sound = sound

        # pre-compute death (out-bounds everywhere) and restart (lily-pads everywhere) states
        obs_vec = DEATH_OBS_VEC
        self.death_state = self.get_state_from_features(obs_vec)
        obs_vec = WIN_OBS_VEC
        self.win_state = self.get_state_from_features(obs_vec)

        # stats
        for i in range(len(ARRIVAL_POSITIONS)):
            self._add_variable(FROGS_ARRIVED_VAR + str(i), StatType.max)
        self._add_variable(NUM_LIVES_VAR, StatType.last)
        self._add_variable(GAME_LEVEL_VAR, StatType.last)
        self._add_variable(GAME_POINTS_VAR, StatType.last)
        self._add_variable(NUM_MOVES_LIFE_VAR, StatType.mean)
        self._add_variable(NUM_DEATHS_VAR, StatType.sum)

        # stats for death cause
        self._add_variable(DEATH_ON_WATER_VAR, StatType.sum)
        self._add_variable(DEATH_BY_CAR_VAR, StatType.sum)
        self._add_variable(DEATH_MAX_MOVES_VAR, StatType.sum)

        # stats for death location
        self._add_variable(DEATH_MIDDLE_GRASS_VAR, StatType.sum)
        self._add_variable(DEATH_BOTTOM_GRASS_VAR, StatType.sum)
        self._add_variable(DEATH_ROAD_VAR, StatType.sum)
        self._add_variable(DEATH_RIVER_VAR, StatType.sum)

        # stats for agent location
        self._add_variable(POS_MIDDLE_GRASS_VAR, StatType.ratio)
        self._add_variable(POS_BOTTOM_GRASS_VAR, StatType.ratio)
        self._add_variable(POS_ROAD_VAR, StatType.ratio)
        self._add_variable(POS_RIVER_VAR, StatType.ratio)

    def register_gym_environment(self, env_id='my-env-v0', display_screen=False, fps=30, show_score_bar=False):
        register(
            id=env_id,
            kwargs={ACTIONS_ATTR: self.config.actions, REWARDS_ATTR: self.config.rewards,
                    LIVES_ATTR: self.config.lives, SPEED_ATTR: self.config.speed,
                    LEVEL_ATTR: self.config.level, NUM_ARRIVED_FROGS_ATTR: self.config.num_arrived_frogs,
                    MAX_STEPS_ATTR: self.config.max_steps_life, DISPLAY_SCREEN_ATTR: display_screen,
                    FPS_ATTR: fps, SHOW_STATS_ATTR: show_score_bar, FORCE_FPS_ATTR: False,
                    SOUND_ATTR: self.sound},
            entry_point=FROGGER_ENTRY_POINT_STR,
            max_episode_steps=self.config.max_steps_per_episode,
            nondeterministic=False,
        )

    def get_state_from_observation(self, obs, rwd, done):
        return self.win_state if self.is_win_state(obs, rwd) else super().get_state_from_observation(obs, rwd, done)

    def get_agent_cell_location(self, obs):

        # converts observation to Frogger game state
        game_state = FroggerState.from_observation(obs)
        fx, fy, *_ = game_state.frog_info
        x = max(0, min(int(fx / CELL_WIDTH), NUM_COLS - 1))
        y = max(0, min(int((fy - MIN_Y_POS) / CELL_HEIGHT), NUM_ROWS - 1))
        return x, y

    def get_cell_coordinates(self, col, row):
        return col * self.config.cell_size[0], MIN_Y_POS + row * self.config.cell_size[1]

    def get_features_from_observation(self, obs, agent_x=-1, agent_y=-1):

        # converts observation to Frogger game state
        game_state = FroggerState.from_observation(obs)
        fx, fy, fw, fh, fd = game_state.frog_info

        # checks whether to use given coordinates
        fx = fx if agent_x == -1 else agent_x
        fy = fy if agent_y == -1 else agent_y

        #         [Left, Right, Up, Down]
        obs_vec = [-1, -1, -1, -1]

        # checks out-of-bounds
        if fx < CELL_WIDTH:
            obs_vec[LEFT_IDX] = OUT_OF_BOUNDS_IDX
        elif fx >= WIDTH - CELL_WIDTH:
            obs_vec[RIGHT_IDX] = OUT_OF_BOUNDS_IDX
        if fy <= MIN_Y_POS:
            obs_vec[UP_IDX] = OUT_OF_BOUNDS_IDX
        elif fy > MAX_Y_POS - CELL_HEIGHT:
            obs_vec[DOWN_IDX] = OUT_OF_BOUNDS_IDX

        # check if frog is near/over water
        if MIN_Y_POS <= fy <= MAX_GRASS_Y_POS:

            # checks thresholds for seeing water
            if obs_vec[UP_IDX] == -1:
                obs_vec[UP_IDX] = WATER_IDX
            if obs_vec[LEFT_IDX] == -1 and fy <= 202:
                obs_vec[LEFT_IDX] = WATER_IDX
            if obs_vec[RIGHT_IDX] == -1 and fy <= 202:
                obs_vec[RIGHT_IDX] = WATER_IDX
            if obs_vec[DOWN_IDX] == - 1 and fy <= 163:
                obs_vec[DOWN_IDX] = WATER_IDX

            # calculates the log delta to anticipate their horizontal movement
            lx_delta = (self.config.speed + game_state.level - self.config.level) * ANIMATIONS_PER_MOVE

            # if agent can jump to a log, it 'sees' it
            on_log_d = -1
            for lx, ly, lw, lh, ld in game_state.log_infos:

                # if agent is in the same line of logs as the log
                if ly <= fy <= ly + lh:
                    if (obs_vec[LEFT_IDX] == -1 or obs_vec[LEFT_IDX] == WATER_IDX) and \
                            lx - 0.25 * fw < fx + 0.5 * fw - CELL_WIDTH < lx + lw + 0.25 * fw:
                        obs_vec[LEFT_IDX] = LOG_IDX
                        # on_log_d = ld
                    if (obs_vec[RIGHT_IDX] == -1 or obs_vec[RIGHT_IDX] == WATER_IDX) and \
                            lx - 0.25 * fw < fx + 0.5 * fw + CELL_WIDTH < lx + lw + 0.25 * fw:
                        obs_vec[RIGHT_IDX] = LOG_IDX
                        # on_log_d = ld

                    # checks if frog is on the log
                    if on_log_d == -1 and lx < fx + fw < lx + lw + fw:
                        on_log_d = ld

            # if frog is on water and not on any log, then its dead
            if on_log_d == -1 and MIN_Y_POS <= fy < MAX_GRASS_Y_POS:
                return DEATH_OBS_VEC

            # if frog is on a log, compensate with log movement
            _fx = fx if on_log_d == -1 else fx + (lx_delta if on_log_d == ACTION_RIGHT_KEY else -lx_delta)

            for lx, ly, lw, lh, ld in game_state.log_infos:
                # ignore same-line logs
                if ly <= fy <= ly + lh:
                    continue

                # if agent is vertically aligned with a log
                if fy < MAX_GRASS_Y_POS:
                    lx += (lx_delta if ld == ACTION_RIGHT_KEY else -lx_delta)
                if lx - fw < _fx <= lx + lw:
                    if (obs_vec[UP_IDX] == -1 or obs_vec[UP_IDX] == WATER_IDX) and \
                            ly <= fy - CELL_HEIGHT <= ly + lh:
                        obs_vec[UP_IDX] = LOG_IDX
                    elif (obs_vec[DOWN_IDX] == -1 or obs_vec[DOWN_IDX] == WATER_IDX) and \
                            ly <= fy + CELL_HEIGHT <= ly + lh:
                        obs_vec[DOWN_IDX] = LOG_IDX

        # checks if frog sees an empty lily pad
        if fy <= MIN_Y_POS:
            fx_delta = self.config.speed + game_state.level - self.config.level
            for i, arrival_x in enumerate(ARRIVAL_POSITIONS):
                if not game_state.arrived_frogs[i] and \
                        arrival_x - ARRIVAL_WIDTH < fx + fx_delta < arrival_x + ARRIVAL_WIDTH:
                    obs_vec[UP_IDX] = LILYPAD_IDX
                    break

        # check if frog is near/over the road
        if MAX_GRASS_Y_POS <= fy <= MAX_Y_POS:

            # calculates the car delta to anticipate their horizontal movement
            cx_delta = (self.config.speed + game_state.level - self.config.level) * 2 * ANIMATIONS_PER_MOVE

            # if a car within vision range, agent 'sees' it
            for cx, cy, cw, ch, cd in game_state.car_infos:

                # if frog is 'within' the car, then its dead
                if cy <= fy <= cy + ch and cx < fx + fw < cx + cw + fw:
                    return DEATH_OBS_VEC

                cx += (cx_delta if cd == ACTION_RIGHT_KEY else -cx_delta)

                # if agent is in the same lane as the car
                if cy <= fy <= cy + ch:
                    car_x_vision = self.config.car_x_vision_num_cells * CELL_WIDTH
                    if (obs_vec[LEFT_IDX] == -1) and fx - car_x_vision <= cx + cw <= fx + fw:
                        obs_vec[LEFT_IDX] = CAR_IDX
                    elif (obs_vec[RIGHT_IDX] == -1) and fx <= cx <= fx + fw + car_x_vision:
                        obs_vec[RIGHT_IDX] = CAR_IDX

                # otherwise, if agent is vertically within the car (according to car direction)
                else:
                    # anticipates car position
                    car_y_tolerance = self.config.car_y_vision_num_cells * CELL_WIDTH
                    # if fx - car_y_tolerance <= cx + cw <= fx + fw + car_y_tolerance + cw:
                    if (cd == ACTION_RIGHT_KEY and fx - car_y_tolerance - cw <= cx <= fx + fw + cx_delta) or \
                            (cd == ACTION_LEFT_KEY and fx - cx_delta <= cx + cw <= fx + fw + car_y_tolerance + cw):
                        if (obs_vec[UP_IDX] == -1) and cy <= fy - CELL_HEIGHT <= cy + ch:
                            obs_vec[UP_IDX] = CAR_IDX
                        elif (obs_vec[DOWN_IDX] == -1) and cy <= fy + CELL_HEIGHT <= cy + ch:
                            obs_vec[DOWN_IDX] = CAR_IDX

        # if agent cannot see any object, sees empty (grass/road)
        for i in range(len(obs_vec)):
            if obs_vec[i] == -1:
                obs_vec[i] = EMPTY_IDX

        return obs_vec

    def get_features_bins(self):
        # num elements in each direction
        return [len(ELEM_LABELS) - 1] * len(DIRS)

    def get_terminal_state(self):
        return self.death_state

    def is_terminal_state(self, obs, rwd, done):
        no_lives_rwd = self.config.rewards[NO_LIVES_RWD_ATTR]
        tick_rwd = self.config.rewards[TICK_RWD_ATTR]
        hit_car_rwd = self.config.rewards[HIT_CAR_RWD_ATTR]
        hit_water_rwd = self.config.rewards[HIT_WATER_RWD_ATTR]

        return rwd == no_lives_rwd + hit_car_rwd + tick_rwd or \
               rwd == no_lives_rwd + hit_water_rwd + tick_rwd or \
               rwd == hit_car_rwd + tick_rwd or \
               rwd == hit_water_rwd + tick_rwd

    def is_win_state(self, obs, rwd):
        # if the agent received frog arrived / new level reward, then its a restart
        arrived_rwd = self.config.rewards[FROG_ARRIVED_RWD_ATTR] if FROG_ARRIVED_RWD_ATTR in self.config.rewards else 0
        new_level_rwd = self.config.rewards[NEW_LEVEL_RWD_ATTR] if NEW_LEVEL_RWD_ATTR in self.config.rewards else 0
        tick_rwd = self.config.rewards[TICK_RWD_ATTR] if TICK_RWD_ATTR in self.config.rewards else 0
        game_state = FroggerState.from_observation(obs)

        return rwd == tick_rwd + arrived_rwd or rwd == tick_rwd + arrived_rwd * game_state.level or \
               (game_state.level > 1 and rwd == tick_rwd + arrived_rwd * (game_state.level - 1)) or \
               rwd == tick_rwd + arrived_rwd + game_state.level * new_level_rwd or \
               (game_state.level > 1 and
                rwd == tick_rwd + arrived_rwd * (game_state.level - 1) + game_state.level * new_level_rwd)

    def get_observation_dissimilarity(self, obs1, obs2):
        # returns difference of points
        return min(1., abs(obs1[2] - obs2[2]) / MAX_POINTS)

    def get_feature_label(self, obs_feat_idx, obs_feat_val):
        return ELEM_LABELS[obs_feat_val]

    def get_features_labels(self, obs_vec, short=False):
        feat_labels = [''] * len(obs_vec)
        for i in range(len(DIRS)):
            feat_labels[i] = '{}: {}'.format(SHORT_DIRS[i] if short else DIRS[i], self.get_feature_label(i, obs_vec[i]))
        return feat_labels

    def print_features(self, obs_vec, columns=False):
        if columns:
            for i in range(len(DIRS)):
                print(DIRS[i].upper().ljust(15), end='\t')
            print()
            for i in range(len(DIRS)):
                print(self.get_feature_label(i, obs_vec[i]).ljust(15), end='\t')
            print()
        else:
            for i in range(len(DIRS)):
                print('Element {}: {}'.format(DIRS[i], self.get_feature_label(i, obs_vec[i])))

    def get_features_image(self, obs_vec):

        # creates pygame surface
        tile_width = self.config.obs_tile_size[0]
        tile_height = self.config.obs_tile_size[1]
        pygame.init()
        surf = pygame.Surface([3 * tile_width, 3 * tile_height], pygame.SRCALPHA, 32)

        # to allow running from any directory
        _dir = join(dirname(abspath(join(__file__, '.'))), 'images')

        # agent goes in the center
        surf.blit(self._load_img(_dir, 'agent'), (tile_width, tile_height, tile_width, tile_height))

        # draws images corresponding to the feature in each direction
        img_file = self.get_feature_label(LEFT_IDX, obs_vec[LEFT_IDX])
        surf.blit(self._load_img(_dir, img_file), (0, tile_height, tile_width, tile_height))
        img_file = self.get_feature_label(RIGHT_IDX, obs_vec[RIGHT_IDX])
        surf.blit(self._load_img(_dir, img_file), (2 * tile_width, tile_height, tile_width, tile_height))
        img_file = self.get_feature_label(UP_IDX, obs_vec[UP_IDX])
        surf.blit(self._load_img(_dir, img_file), (tile_width, 0, tile_width, tile_height))
        img_file = self.get_feature_label(DOWN_IDX, obs_vec[DOWN_IDX])
        surf.blit(self._load_img(_dir, img_file), (tile_width, 2 * tile_height, tile_width, tile_height))

        return surf

    def get_known_goal_states(self):

        # gets all combinations water/log/out-bounds in 3 directions (except up)
        adj_combs = list(product([WATER_IDX, LOG_IDX, OUT_OF_BOUNDS_IDX], repeat=3))

        # gets all states where lilypad is up to the agent
        goal_states = []
        for adj_coin_comb in adj_combs:
            obs_vec = [adj_coin_comb[0], adj_coin_comb[1], LILYPAD_IDX, adj_coin_comb[2]]
            goal_states.append(self.get_state_from_features(obs_vec))

        return goal_states

    def get_known_feature_action_assocs(self):
        feat_action_assocs = []

        # if the agent sees a lilypad up, go in that direction
        feat_action_assocs.append((UP_IDX, LILYPAD_IDX, UP_IDX))

        # if the agent sees log up, jump to it
        feat_action_assocs.append((UP_IDX, LOG_IDX, UP_IDX))

        # if the agent sees empty up, move up
        feat_action_assocs.append((UP_IDX, EMPTY_IDX, UP_IDX))

        # if the agent sees a car left/right try to avoid it
        feat_action_assocs.append((LEFT_IDX, CAR_IDX, RIGHT_IDX))
        feat_action_assocs.append((RIGHT_IDX, CAR_IDX, LEFT_IDX))

        return feat_action_assocs

    def update_stats(self, e, t, obs, n_obs, s, a, r, ns):
        super().update_stats(e, t, obs, n_obs, s, a, r, ns)

        game_state = FroggerState.from_observation(n_obs)
        self.stats_collector.add_sample(GAME_POINTS_VAR, e, game_state.points)
        for i in range(len(ARRIVAL_POSITIONS)):
            self.stats_collector.add_sample(FROGS_ARRIVED_VAR + str(i), e, int(game_state.arrived_frogs[i]))
        self.stats_collector.add_sample(NUM_LIVES_VAR, e, game_state.lives)
        self.stats_collector.add_sample(GAME_LEVEL_VAR, e, game_state.level)
        self.stats_collector.add_sample(NUM_MOVES_LIFE_VAR, e, self.config.max_steps_life - game_state.steps)
        self.stats_collector.add_sample(NUM_DEATHS_VAR, e, 0 if game_state.death_idx == NOT_DEAD_IDX else 1)

        # checks death type
        if game_state.death_idx == NOT_DEAD_IDX:
            self.stats_collector.add_sample(DEATH_ON_WATER_VAR, e, 0)
            self.stats_collector.add_sample(DEATH_BY_CAR_VAR, e, 0)
            self.stats_collector.add_sample(DEATH_MAX_MOVES_VAR, e, 0)
        elif game_state.death_idx == CAR_DEATH_IDX:
            self.stats_collector.add_sample(DEATH_BY_CAR_VAR, e, 1)
        elif game_state.death_idx == WATER_DEATH_IDX:
            self.stats_collector.add_sample(DEATH_ON_WATER_VAR, e, 1)
        elif game_state.death_idx == TIME_UP_DEATH_IDX:
            self.stats_collector.add_sample(DEATH_MAX_MOVES_VAR, e, 1)

        # checks agent location (and death)
        fy = FroggerState.from_observation(obs).frog_info[1]
        if fy == MAX_GRASS_Y_POS:
            self.stats_collector.add_sample(POS_MIDDLE_GRASS_VAR, e, 1)
            self.stats_collector.add_sample(POS_ROAD_VAR, e, 0)
            self.stats_collector.add_sample(POS_RIVER_VAR, e, 0)
            self.stats_collector.add_sample(POS_BOTTOM_GRASS_VAR, e, 0)

            if game_state.death_idx != NOT_DEAD_IDX and game_state.death_idx != TIME_UP_DEATH_IDX:
                self.stats_collector.add_sample(DEATH_MIDDLE_GRASS_VAR, e, 1)
            self.stats_collector.add_sample(DEATH_ROAD_VAR, e, 0)
            self.stats_collector.add_sample(DEATH_RIVER_VAR, e, 0)
            self.stats_collector.add_sample(DEATH_BOTTOM_GRASS_VAR, e, 0)

        elif MIN_Y_POS <= fy < MAX_GRASS_Y_POS:
            self.stats_collector.add_sample(POS_RIVER_VAR, e, 1)
            self.stats_collector.add_sample(POS_MIDDLE_GRASS_VAR, e, 0)
            self.stats_collector.add_sample(POS_ROAD_VAR, e, 0)
            self.stats_collector.add_sample(POS_BOTTOM_GRASS_VAR, e, 0)

            if game_state.death_idx != NOT_DEAD_IDX and game_state.death_idx != TIME_UP_DEATH_IDX:
                self.stats_collector.add_sample(DEATH_RIVER_VAR, e, 1)
            self.stats_collector.add_sample(DEATH_MIDDLE_GRASS_VAR, e, 0)
            self.stats_collector.add_sample(DEATH_ROAD_VAR, e, 0)
            self.stats_collector.add_sample(DEATH_BOTTOM_GRASS_VAR, e, 0)

        elif MAX_GRASS_Y_POS < fy < MAX_Y_POS:
            self.stats_collector.add_sample(POS_ROAD_VAR, e, 1)
            self.stats_collector.add_sample(POS_MIDDLE_GRASS_VAR, e, 0)
            self.stats_collector.add_sample(POS_RIVER_VAR, e, 0)
            self.stats_collector.add_sample(POS_BOTTOM_GRASS_VAR, e, 0)

            if game_state.death_idx != NOT_DEAD_IDX and game_state.death_idx != TIME_UP_DEATH_IDX:
                self.stats_collector.add_sample(DEATH_ROAD_VAR, e, 1)
            self.stats_collector.add_sample(DEATH_MIDDLE_GRASS_VAR, e, 0)
            self.stats_collector.add_sample(DEATH_RIVER_VAR, e, 0)
            self.stats_collector.add_sample(DEATH_BOTTOM_GRASS_VAR, e, 0)

        elif fy == MAX_Y_POS:
            self.stats_collector.add_sample(POS_BOTTOM_GRASS_VAR, e, 1)
            self.stats_collector.add_sample(POS_MIDDLE_GRASS_VAR, e, 0)
            self.stats_collector.add_sample(POS_ROAD_VAR, e, 0)
            self.stats_collector.add_sample(POS_RIVER_VAR, e, 0)

            if game_state.death_idx != NOT_DEAD_IDX and game_state.death_idx != TIME_UP_DEATH_IDX:
                self.stats_collector.add_sample(DEATH_BOTTOM_GRASS_VAR, e, 1)
            self.stats_collector.add_sample(DEATH_MIDDLE_GRASS_VAR, e, 0)
            self.stats_collector.add_sample(DEATH_ROAD_VAR, e, 0)
            self.stats_collector.add_sample(DEATH_RIVER_VAR, e, 0)

    def update_stats_episode(self, e, path=None):
        if (e + 1) % (self.config.num_episodes / 100) == 0:
            clean_console()
            print('Episode {} ({:.0f}%)...'.format(e + 1, ((e + 1) / self.config.num_episodes) * 100))
            self._print_stats(e, PRINT_SCREEN_VAR_NAMES)

    def save_stats(self, path, clear=True, img_format='pdf'):
        super().save_stats(path, clear)

        # collects and prints final stats to file
        e = self.config.num_episodes
        with open(join(path, 'results.log'), 'w') as file:
            print_line('\nStats (avg. of {} episodes):'.format(e), file)
            var_names = list(self.stats_collector.all_variables())
            var_names.sort()
            self._print_stats(e, var_names, file)

    def save_state_features(self, out_file, delimiter=CSV_DELIMITER):
        feats_nbins = self.get_features_bins()
        num_elems = len(ELEM_LABELS) - 1
        num_states = self.config.num_states
        data = [None] * num_states

        for l in range(num_elems):
            for r in range(num_elems):
                for u in range(num_elems):
                    for d in range(num_elems):
                        # gets discretized index
                        obs_vec = [l, r, u, d]
                        state = get_discretized_index(obs_vec, feats_nbins)

                        # puts element names in correct place in table
                        data[state] = [state,
                                       self.get_feature_label(0, l),
                                       self.get_feature_label(1, r),
                                       self.get_feature_label(2, u),
                                       self.get_feature_label(3, d)]

        header = [''] * 5
        header[0] = 'state'
        for i in range(len(DIRS)):
            header[i + 1] = 'element {}'.format(DIRS[i])

        # saves table
        df = pd.DataFrame(data, columns=header)
        df.to_csv(out_file, delimiter, index=False)

    @staticmethod
    def _load_img(base_dir, img_file):
        return pygame.image.load(join(base_dir, '{}.png'.format(img_file)))
