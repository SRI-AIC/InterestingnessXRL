__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

from os.path import join
from interestingness_xrl.learning.explorations import ExpDecaySoftMax, SoftMaxExploration, ManualExploration, \
    ExplorationStrategy
from interestingness_xrl.learning.agents import QLearningAgent, QValueBasedAgent, RandomAgent
from interestingness_xrl.scenarios.frogger.configurations import *
from interestingness_xrl.scenarios.frogger.scenario_helper import FroggerHelper
from interestingness_xrl.explainability.explanation.heatmaps import HeatmapsExplainer
from interestingness_xrl.explainability.explanation.highlights import HighlightsExplainer
from interestingness_xrl.explainability.explanation.sequences import SequencesExplainer
from interestingness_xrl.explainability.explanation.tracker import AspectsTrackerExplainer

HIGHLIGHTS_FPS = 16  # 20  # 12
HIGHLIGHT_TIME_STEPS = 21  # 10 before and 10 after the state/state-action pair
SEQUENCE_ADDITIONAL_TIME_STEPS = 6  # to be added before and after sequence
MAX_HIGHLIGHTS_PER_ASPECT = 4
MAX_BETWEEN_STATES = 50  # number of allowed non-sequence states between 2 states in a sequence

# DEFAULT_CONFIG = FROGGER_CONFIG
# DEFAULT_CONFIG = FROGGER_LIMITED_CONFIG
# DEFAULT_CONFIG = FROGGER_HIGH_VISION_CONFIG
DEFAULT_CONFIG = FROGGER_CONFIG
# DEFAULT_CONFIG = FAST_FROGGER_CONFIG

ANALYSIS_CONFIGS = {
    FROGGER_CONFIG: FROGGER_ANALYSIS_CONFIG,
    FROGGER_LIMITED_CONFIG: FROGGER_ANALYSIS_CONFIG,
    FROGGER_HIGH_VISION_CONFIG: FROGGER_ANALYSIS_CONFIG,
}

DATA_ORGANIZER_CONFIGS = OrderedDict([
    ('Optimized', FROGGER_CONFIG),
    # ('Short-vision', FROGGER_LIMITED_CONFIG),
    ('High-vision', FROGGER_HIGH_VISION_CONFIG),
    ('Fear-water', FROGGER_FEAR_WATER_CONFIG)
])


class ReactiveStrategy(ExplorationStrategy):
    """
    Corresponds to a reactive strategy, i.e., that selects actions based on the information from the state.
    """

    def __init__(self, helper, rng=None):
        """
        Creates a new reactive strategy according to the given arguments.
        :param ScenarioHelper helper: the helper containing the reactive strategy.
        :param np.random.RandomState rng: the random number generator to be used by the selection strategy.
        """
        super().__init__(rng)
        self.helper = helper

    def explore(self, state):
        return self.helper.act_reactive(state, self.rng)


class AgentType(object):
    """
    Contains definitions for all types of agent that can be run in the simulations.
    """
    Learning = 0
    Testing = 1
    Random = 2
    Reactive = 3
    Manual = 4

    @staticmethod
    def get_name(agent_t):
        if agent_t == AgentType.Learning:
            return 'learn'
        if agent_t == AgentType.Testing:
            return 'test'
        if agent_t == AgentType.Random:
            return 'random'
        if agent_t == AgentType.Reactive:
            return 'reactive'
        if agent_t == AgentType.Manual:
            return 'manual'
        return 'Unknown'


class ReportType(object):
    """
    Contains definitions for all types of explanation that can be performed.
    """
    Highlights = 0
    Sequences = 1
    Heatmaps = 2
    AspectsTracker = 3

    @staticmethod
    def get_name(explanation_t):
        if explanation_t == ReportType.Heatmaps:
            return 'heatmaps'
        if explanation_t == ReportType.Highlights:
            return 'highlights'
        if explanation_t == ReportType.Sequences:
            return 'sequences'
        if explanation_t == ReportType.AspectsTracker:
            return 'tracker'
        return 'Unknown'


def create_helper(config, sound=False):
    """
    Creates a scenario helper according to the given environment configuration.
    :param EnvironmentConfiguration config: the environment configuration.
    :param bool sound: whether to play sounds.
    :rtype: ScenarioHelper
    :return: a helper containing all necessary methods to run a simulation scenario.
    """
    if isinstance(config, FroggerConfiguration):
        return FroggerHelper(config, sound=sound)


def create_agent(helper, agent_t, rng):
    """
    Creates an agent and exploration strategy according to the given parameters.
    :param ScenarioHelper helper: the helper containing all necessary methods to run a simulation scenario.
    :param int agent_t: the type of agent to be created.
    :param np.random.RandomState rng: the random number generator to be used by the action selection strategy.
    :rtype: tuple
    :return: a tuple (agent, exploration_strat) containing the created agent and respective exploration strategy.
    """
    config = helper.config
    agent = None
    exploration_strategy = None

    # learning: Q-learning with decreasing SoftMax
    if agent_t == AgentType.Learning:
        exploration_strategy = ExpDecaySoftMax(config.num_episodes, config.max_temp, config.min_temp, rng)
        agent = QLearningAgent(config.num_states, config.num_actions, True, config.get_action_names(),
                               config.learn_rate, config.discount, config.initial_q_value, exploration_strategy)

    # testing: Q-agent (table loaded from learning) with fixed (greedy) SoftMax
    if agent_t == AgentType.Testing:
        exploration_strategy = SoftMaxExploration(config.min_temp, rng)
        agent = QValueBasedAgent(config.num_states, config.num_actions,
                                 action_names=config.get_action_names(), exploration_strategy=exploration_strategy)

    # random agent
    if agent_t == AgentType.Random:
        agent = RandomAgent(config.num_states, config.num_actions, True, config.get_action_names(), rng)
        exploration_strategy = agent.exploration_strategy

    # reactive agent uses known strategy
    if agent_t == AgentType.Reactive:
        exploration_strategy = ReactiveStrategy(helper, rng)
        agent = QValueBasedAgent(config.num_states, config.num_actions,
                                 action_names=config.get_action_names(), exploration_strategy=exploration_strategy)

    # manual agent: Q-learning with manually-selected actions
    if agent_t == AgentType.Manual:
        exploration_strategy = ManualExploration()
        agent = QLearningAgent(config.num_states, config.num_actions, True, config.get_action_names(),
                               config.learn_rate, config.discount, config.initial_q_value, exploration_strategy)

    # assigns agent to helper for collecting stats
    helper.agent = agent

    return agent, exploration_strategy


def _get_base_dir(config):
    """
    Gets a path to the root of the results directory for a given environment/scenario.
    :param EnvironmentConfiguration config: the environment configuration.
    :rtype: str
    :return: the path to the root of the results directory.
    """
    return join('results', config.name)


def get_agent_output_dir(config, agent_t, trial_num=0):
    """
    Gets a path to the root of the results directory for the given type of agent and environment/scenario.
    :param EnvironmentConfiguration config: the environment configuration.
    :param int agent_t: the type of agent to be created.
    :param int trial_num: the number of the trial
    :rtype: str
    :return: the path to the root of the agent's results directory.
    """
    return join(_get_base_dir(config), AgentType.get_name(agent_t), str(trial_num))


def get_observations_output_dir(agent_dir):
    """
    Gets a path to the directory containing the observations for the given type of agent and environment/scenario.
    :param str agent_dir: the directory containing the agent's results.
    :rtype: str
    :return: the path to the observations directory.
    """
    return join(agent_dir, 'observations')


def get_analysis_output_dir(agent_dir):
    """
    Gets a path to the directory containing the analysis for the given type of agent and environment/scenario.
    :param str agent_dir: the directory containing the agent's results.
    :rtype: str
    :return: the path to the analysis directory.
    """
    return join(agent_dir, 'analysis')


def get_analysis_config(config):
    """
    Gets the analysis configuration corresponding to the given environment configuration.
    :param EnvironmentConfiguration config: the environment configuration.
    :rtype: AnalysisConfiguration
    :return: the analysis configuration corresponding to the given environment configuration.
    """
    if isinstance(config, FroggerConfiguration):
        return FROGGER_ANALYSIS_CONFIG

    return None


def get_explanation_output_dir(agent_dir, explanation_t):
    """
    Gets a path to the directory containing the explanations for the given type of agent and explanation type.
    :param str agent_dir: the directory containing the agent's results.
    :param int explanation_t: the type of explanation.
    :rtype: str
    :return: the path to the explanation directory.
    """
    return join(agent_dir, ReportType.get_name(explanation_t))


def create_explainer(explanation_t, env, helper, full_analysis, output_dir, recorded_episodes):
    """
    Creates a new explainer according to the provided parameters.
    :param int explanation_t: the type of explainer to create.
    :param Env env: the Gym environment to be tracked, from which the frames are extracted.
    :param ScenarioHelper helper: the helper containing all necessary methods to extract information from the env.
    :param FullAnalysis full_analysis: the full analysis over the agent's history of interaction with the environment.
    :param str output_dir: the path to the output directory in which to save the videos.
    :param list recorded_episodes: the episodes in which episodes are to be recorded.
    :rtype: Explainer
    :return: the created explainer.
    """
    if explanation_t == ReportType.Heatmaps:
        return HeatmapsExplainer(env, helper, full_analysis, output_dir, recorded_episodes)

    if explanation_t == ReportType.Highlights:
        return HighlightsExplainer(env, helper, full_analysis, output_dir, recorded_episodes,
                                   HIGHLIGHTS_FPS, HIGHLIGHT_TIME_STEPS, MAX_HIGHLIGHTS_PER_ASPECT)

    if explanation_t == ReportType.Sequences:
        return SequencesExplainer(env, helper, full_analysis, output_dir, recorded_episodes,
                                  HIGHLIGHTS_FPS, SEQUENCE_ADDITIONAL_TIME_STEPS, MAX_HIGHLIGHTS_PER_ASPECT,
                                  MAX_BETWEEN_STATES)

    if explanation_t == ReportType.AspectsTracker:
        return AspectsTrackerExplainer(env, helper, full_analysis, output_dir, recorded_episodes, HIGHLIGHTS_FPS)
