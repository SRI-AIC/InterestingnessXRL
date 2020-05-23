__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

from collections import OrderedDict
from interestingness_xrl.scenarios.configurations import EnvironmentConfiguration, AnalysisConfiguration
from frogger import ACTION_UP_KEY, ACTION_DOWN_KEY, ACTION_LEFT_KEY, ACTION_RIGHT_KEY, HIT_CAR_RWD_ATTR, \
    HIT_WATER_RWD_ATTR, TIME_UP_RWD_ATTR, NEW_LEVEL_RWD_ATTR, FROG_ARRIVED_RWD_ATTR, TICK_RWD_ATTR, NO_LIVES_RWD_ATTR, \
    CELL_WIDTH, CELL_HEIGHT, NUM_COLS, NUM_ROWS

GAME_GYM_ID = 'Frogger-Custom'


class FroggerConfiguration(EnvironmentConfiguration):
    """
    A class used to store configurations of tests / learning simulations on the Frogger game.
    """

    def __init__(self, name, actions, rewards, gym_env_id, lives=3, speed=3., level=1, num_arrived_frogs=5,
                 max_steps_life=60, car_x_vision_num_cells=2., car_y_vision_num_cells=1.,
                 max_steps_per_episode=300, num_episodes=1000, num_recorded_videos=10, seed=0, max_temp=20,
                 min_temp=0.05, discount=.9, learn_rate=.3, initial_q_value=0.):
        """
        Creates a new configuration with the given parameters.
        :param str name: the name of the configuration.
        :param OrderedDict actions: the actions available for the agent in a 'action_name : [keyboard_codes]' fashion.
        :param dict rewards: the reward function in an 'element_name : value' fashion.
        :param str gym_env_id: the name identifier for the gym environment.
        :param int lives: the number of lives available to the agent.
        :param float speed: the initial speed of the game (cars and logs).
        :param int level: the initial level of the game.
        :param int num_arrived_frogs: the number of frogs that have to be crossed to end the game.
        :param int max_steps_life: the maximum number of steps per life (for the frog to reach a lily pad).
        :param float car_x_vision_num_cells: the number of cells away that the agent can detect a car on its left / right.
        :param float car_y_vision_num_cells: the number of cells away from a car so that the agent can detect its presence up /down.
        :param int max_steps_per_episode: the maximum number of steps in one episode.
        :param int num_episodes: the number of episodes used to train/test the agent.
        :param int num_recorded_videos: the number of videos to record during the test episodes.
        :param int seed: the seed used for the random number generator used by the agent.
        :param float max_temp: the maximum temperature of the Soft-max action-selection strategy (start of training).
        :param float min_temp: the minimum temperature of the Soft-max action-selection strategy (end of training).
        :param float discount: the discount factor in [0, 1] (how important are the future rewards?).
        :param float learn_rate: the agent's learning rate (the weight associated to a new sample during learning).
        :param float initial_q_value: the value used to initialize the q-function (e.g., for optimistic initialization).
        """
        # 6 elements (grass, water, car, log, lilypad, out-bounds)  ^ 4 dirs
        num_states = 6 ** 4

        super().__init__(name, num_states, actions, rewards, gym_env_id, max_steps_per_episode, num_episodes,
                         num_recorded_videos, seed, max_temp, min_temp, discount, learn_rate, initial_q_value,
                         (CELL_WIDTH, CELL_HEIGHT), (NUM_COLS, NUM_ROWS), (60, 60))

        self.lives = lives
        self.speed = speed
        self.level = level
        self.num_arrived_frogs = num_arrived_frogs
        self.max_steps_life = max_steps_life
        self.car_x_vision_num_cells = car_x_vision_num_cells
        self.car_y_vision_num_cells = car_y_vision_num_cells


FROGGER_CONFIG = FroggerConfiguration(
    name='frogger',
    actions=OrderedDict([
        ('left', ACTION_LEFT_KEY),
        ('right', ACTION_RIGHT_KEY),
        ('up', ACTION_UP_KEY),
        ('down', ACTION_DOWN_KEY),
        # ('nowhere', ACTION_NO_MOVE_KEY)
    ]),
    rewards={
        HIT_CAR_RWD_ATTR: -200.,
        HIT_WATER_RWD_ATTR: -200.,
        TIME_UP_RWD_ATTR: -200.,
        NO_LIVES_RWD_ATTR: -300.,
        NEW_LEVEL_RWD_ATTR: 0.,
        FROG_ARRIVED_RWD_ATTR: 5000.,
        TICK_RWD_ATTR: -1.
    },
    gym_env_id=GAME_GYM_ID,
    lives=3,
    speed=3.,
    level=1,
    num_arrived_frogs=2,
    car_x_vision_num_cells=1.5,
    car_y_vision_num_cells=1.,
    max_steps_life=100,
    max_steps_per_episode=300,
    num_episodes=2000,
    num_recorded_videos=10,
    seed=0,
    max_temp=20,
    min_temp=0.05,
    discount=.9,
    learn_rate=.3,
    initial_q_value=5000.
)

FROGGER_LIMITED_CONFIG = FroggerConfiguration(
    name='frogger-limited-vision',
    actions=OrderedDict([
        ('left', ACTION_LEFT_KEY),
        ('right', ACTION_RIGHT_KEY),
        ('up', ACTION_UP_KEY),
        ('down', ACTION_DOWN_KEY),
        # ('nowhere', ACTION_NO_MOVE_KEY)
    ]),
    rewards={
        HIT_CAR_RWD_ATTR: -200.,
        HIT_WATER_RWD_ATTR: -200.,
        TIME_UP_RWD_ATTR: -200.,
        NO_LIVES_RWD_ATTR: -300.,
        NEW_LEVEL_RWD_ATTR: 0.,
        FROG_ARRIVED_RWD_ATTR: 5000.,
        TICK_RWD_ATTR: -1.
    },
    gym_env_id=GAME_GYM_ID,
    lives=3,
    speed=3.,
    level=1,
    num_arrived_frogs=2,
    car_x_vision_num_cells=.0,  # changed
    car_y_vision_num_cells=.0,  # changed
    max_steps_life=100,
    max_steps_per_episode=300,
    num_episodes=2000,
    num_recorded_videos=10,
    seed=0,
    max_temp=20,
    min_temp=0.05,
    discount=.9,
    learn_rate=.3,
    initial_q_value=5000.
)

FROGGER_HIGH_VISION_CONFIG = FroggerConfiguration(
    name='frogger-high-vision',
    actions=OrderedDict([
        ('left', ACTION_LEFT_KEY),
        ('right', ACTION_RIGHT_KEY),
        ('up', ACTION_UP_KEY),
        ('down', ACTION_DOWN_KEY),
        # ('nowhere', ACTION_NO_MOVE_KEY)
    ]),
    rewards={
        HIT_CAR_RWD_ATTR: -200.,
        HIT_WATER_RWD_ATTR: -200.,
        TIME_UP_RWD_ATTR: -200.,
        NO_LIVES_RWD_ATTR: -300.,
        NEW_LEVEL_RWD_ATTR: 0.,
        FROG_ARRIVED_RWD_ATTR: 5000.,
        TICK_RWD_ATTR: -1.
    },
    gym_env_id=GAME_GYM_ID,
    lives=3,
    speed=3.,
    level=1,
    num_arrived_frogs=2,
    car_x_vision_num_cells=3.5,  # changed
    car_y_vision_num_cells=3.0,  # changed
    max_steps_life=100,
    max_steps_per_episode=300,
    num_episodes=2000,
    num_recorded_videos=10,
    seed=0,
    max_temp=20,
    min_temp=0.05,
    discount=.9,
    learn_rate=.3,
    initial_q_value=5000.
)

FROGGER_FEAR_WATER_CONFIG = FroggerConfiguration(
    name='frogger-fear-water',
    actions=OrderedDict([
        ('left', ACTION_LEFT_KEY),
        ('right', ACTION_RIGHT_KEY),
        ('up', ACTION_UP_KEY),
        ('down', ACTION_DOWN_KEY),
        # ('nowhere', ACTION_NO_MOVE_KEY)
    ]),
    rewards={
        HIT_CAR_RWD_ATTR: -200.,
        HIT_WATER_RWD_ATTR: -10000,  # changed
        TIME_UP_RWD_ATTR: -200.,
        NO_LIVES_RWD_ATTR: -300.,
        NEW_LEVEL_RWD_ATTR: 0.,
        FROG_ARRIVED_RWD_ATTR: 5000.,
        TICK_RWD_ATTR: -1.
    },
    gym_env_id=GAME_GYM_ID,
    lives=3,
    speed=3.,
    level=1,
    num_arrived_frogs=2,
    car_x_vision_num_cells=1.5,
    car_y_vision_num_cells=1.,
    max_steps_life=100,
    max_steps_per_episode=300,
    num_episodes=2000,
    num_recorded_videos=10,
    seed=0,
    max_temp=20,
    min_temp=0.05,
    discount=.9,
    learn_rate=.3,
    initial_q_value=0.  # changed
)

FAST_FROGGER_CONFIG = FroggerConfiguration(
    name='frogger-fast',
    actions=OrderedDict([
        ('left', ACTION_LEFT_KEY),
        ('right', ACTION_RIGHT_KEY),
        ('up', ACTION_UP_KEY),
        ('down', ACTION_DOWN_KEY),
    ]),
    rewards={
        HIT_CAR_RWD_ATTR: -200.,
        HIT_WATER_RWD_ATTR: -200.,
        TIME_UP_RWD_ATTR: 0.,
        NO_LIVES_RWD_ATTR: 0.,
        NEW_LEVEL_RWD_ATTR: 0.,
        FROG_ARRIVED_RWD_ATTR: 10000.,
        TICK_RWD_ATTR: 0.
    },
    gym_env_id=GAME_GYM_ID,
    lives=3,
    speed=3.,
    level=1,
    num_arrived_frogs=2,
    car_x_vision_num_cells=1.5,
    car_y_vision_num_cells=1.,
    max_steps_life=100,
    max_steps_per_episode=300,
    num_episodes=200,  # changed
    num_recorded_videos=1,  # changed
    seed=0,
    max_temp=20,
    min_temp=0.05,
    discount=.9,
    learn_rate=.6,  # changed
    initial_q_value=500.
)

FROGGER_ANALYSIS_CONFIG = AnalysisConfiguration(
    min_count=7,
    trans_min_states=5,
    certain_trans_max_disp=0.03,
    uncertain_trans_min_disp=0.9,
    rwd_outlier_stds=2.5,
    freq_min_state_count=15000,
    infreq_max_state_count=150,
    min_feat_set_count=15000,
    assoc_min_feat_set_jacc=0.4,  # todo
    assoc_min_feat_rule_conf=0.8,
    no_assoc_max_feat_set_jacc=0.0001,
    certain_exec_max_disp=0.1,
    uncertain_exec_min_disp=0.85,
    value_outlier_stds=2,
    pred_error_outlier_stds=2.,
    max_time_step=0.4,  # todo
    val_diff_var_outlier_stds=2,
    action_jsd_threshold=0.16
)
