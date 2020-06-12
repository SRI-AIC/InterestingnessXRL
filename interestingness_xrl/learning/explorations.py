__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
from abc import abstractmethod, ABC


class ExplorationStrategy(ABC):
    """
    Represents an exploration-exploitation action-selection strategy to be used by agents during learning.
    """

    def __init__(self, rng=None):
        """
        Creates a new exploration strategy according to the given arguments.
        :param np.random.RandomState rng: the random number generator to be used by the selection strategy.
        """
        self.rng = np.random.RandomState(0) if rng is None else rng
        self.agent = None

    @abstractmethod
    def explore(self, state):
        """
        Selects an action to be executed according to the given state.
        :param int state: the index of the environment's state as perceived by the agent.
        :rtype: int
        :return: the index of the action to be executed.
        """
        pass

    @abstractmethod
    def update(self, num_episode):
        """
        Updates the exploration strategy according to the number of training episodes.
        :param int num_episode: the number of the current training episode.
        :return:
        """
        pass


class ManualExploration(ExplorationStrategy, ABC):
    """
    Represents a manual exploration strategy where , i.e., selecting actions as given by some external provider.
    """

    def __init__(self):
        """
        Creates a new manual exploration strategy.
        """
        super().__init__()
        self.action = 0

    def set_action(self, a):
        """
        Sets the action to be executed.
        :param int a: the index of the action to be executed.
        :return:
        """
        self.action = a

    def explore(self, state):
        return self.action


class GreedyExploration(ExplorationStrategy, ABC):
    """
    Represents a greedy exploration strategy, i.e., one that selects an action among the ones with the highest Q-value.
    """

    def __init__(self, rng=None):
        """
        Creates a new exploration strategy according to the given arguments.
        :param np.random.RandomState rng: the random number generator to be used by the selection strategy.
        """
        super().__init__(rng)

    def explore(self, state):
        """
        Selects an action among the ones with the highest Q-value for the given state, at random.
        :param int state: the index of the environment's state as perceived by the agent.
        :return int: the index of the action to be executed.
        """
        q = self.agent.q[state]
        return self.rng.choice(np.flatnonzero(q == q.max()))


class RandomExploration(ExplorationStrategy):
    """
    Represents a random exploration strategy, i.e., one that selects actions at random, independently of the state.
    """

    def __init__(self, rng=None):
        """
        Creates a new exploration strategy according to the given arguments.
        :param np.random.RandomState rng: the random number generator to be used by the selection strategy.
        """
        super().__init__(rng)

    def explore(self, state):
        """
        Selects an action at random, independently of the state.
        :param int state: the index of the environment's state as perceived by the agent.
        :return int: the index of the action to be executed.
        """
        return self.rng.randint(0, self.agent.num_actions)


class EpsilonGreedyExploration(ExplorationStrategy, ABC):
    """
    Represents a greedy exploration strategy using epsilon-greedy action-selection, i.e., selects an action at random
    with probability epsilon, and an action among the ones with the highest Q-value with probability 1 - epsilon.
    """

    def __init__(self, eps=.2, rng=None):
        """
        Creates a new exploration strategy according to the given arguments.
        :param float eps: the probability in [0, 1] with which to select a random action vs. a greedy action.
        :param np.random.RandomState rng: the random number generator to be used by the selection strategy.
        """
        super().__init__(rng)
        self.eps = eps

    def explore(self, state):
        """
        Selects an action action at random with probability epsilon, and an action among the ones with the highest
        Q-value with probability 1 - epsilon.
        :param int state: the index of the environment's state as perceived by the agent.
        :return int: the index of the action to be executed.
        """
        q = self.agent.q[state]
        return self.rng.choice(np.flatnonzero(q == q.max())) if self.rng.uniform(0, 1) >= self.eps else \
            self.rng.randint(0, self.agent.num_actions)


class LinearDecreaseEpsilonGreedy(EpsilonGreedyExploration):
    """
    Represents an epsilon-greedy exploration strategy in which the parameter epsilon decreases linearly throughout
    training.
    """

    def __init__(self, max_episodes, start_eps=1., rng=None):
        """
        Creates a new exploration strategy according to the given arguments.
        :param int max_episodes: the maximum number of training episodes.
        :param float start_eps: the initial value of epsilon, i.e., at the start of training.
        :param np.random.RandomState rng: the random number generator to be used by the selection strategy.
        """
        super().__init__(start_eps, rng)
        self.max_episodes = max_episodes
        self.start_eps = start_eps
        self.eps = 0.

    def update(self, num_episode):
        """
        Updates the epsilon parameter of the exploration strategy as a linear function of the episode number, i.e.,
        where epsilon decreases linearly with the number of training episodes.
        :param int num_episode: the number of the current training episode.
        :return:
        """
        self.eps = self.start_eps * max(0., 1. - ((num_episode + 1) / self.max_episodes))


class ExpDecayEpsilonGreedy(LinearDecreaseEpsilonGreedy):
    """
    Represents an epsilon-greedy exploration strategy in which the parameter epsilon decreases exponentially throughout
    training.
    """

    def __init__(self, max_episodes, start_eps=1., rng=None):
        """
        Creates a new exploration strategy according to the given arguments.
        :param int max_episodes: the maximum number of training episodes.
        :param float start_eps: the initial value of epsilon, i.e., at the start of training.
        :param np.random.RandomState rng: the random number generator to be used by the selection strategy.
        """
        super().__init__(max_episodes, start_eps, rng)
        self.eps = 0.

    def update(self, num_episode):
        """
        Updates the epsilon parameter of the exploration strategy as an exponential function of the episode number,
        i.e., where epsilon decreases exponentially with the number of training episodes.
        :param int num_episode: the number of the current training episode.
        :return:
        """
        self.eps = self.start_eps * max(0., (1 - (10 / self.max_episodes)) ** num_episode)


class AdaptiveEpsilonGreedy(EpsilonGreedyExploration):
    """
    Represents a Value-Difference Based Exploration (VBDE) epsilon-greedy exploration strategy that adapts the
    exploration parameter epsilon according to the temporal-difference error observed from value-function backups.
    See: https://doi.org/10.1007/978-3-642-16111-7_23
    """

    def __init__(self, inverse_sensitivity=1., rng=None):
        """
        Creates a new exploration strategy according to the given arguments.
        :param float inverse_sensitivity: positive constant controlling for how TD influences exploration.
        :param np.random.RandomState rng: the random number generator to be used by the selection strategy.
        """
        super().__init__(1., rng)
        self.inverse_sensitivity = inverse_sensitivity
        self.eps = 0.

    def explore(self, state):
        self.eps = 1 + np.exp(-np.mean(self.agent.dq[state]) / self.inverse_sensitivity)
        return super().explore(state)

    def update(self, num_episode):
        pass


class SoftMaxExploration(ExplorationStrategy):
    """
    Represents an exploration strategy using a soft-max action-selection, i.e., where actions are selected according to
    a Gibbs (or Boltzmann) distribution defined by the Q-values of the actions and a temperature parameter.
    High temperatures cause the actions to be selected with (nearly) the same probability. Low temperatures cause a
    greater difference in selection probability for actions that differ in their Q-value estimates. As the temperature
    approaches 0, the action selection becomes the same as the greedy exploration.
    """

    def __init__(self, temp=20, rng=None):
        """
        Creates a new exploration strategy according to the given arguments.
        :param float temp: the temperature parameter.
        :param np.random.RandomState rng: the random number generator to be used by the selection strategy.
        """
        super().__init__(rng)
        self.temp = temp

    def explore(self, state):
        """
        Selects an action action according to a Gibbs (or Boltzmann) distribution defined by the Q-values of the actions
        and a temperature parameter.
        :param int state: the index of the environment's state as perceived by the agent.
        :return int: the index of the action to be executed.
        """
        q = self.agent.q[state]

        # calculates normalization factor
        q_max = np.max(np.abs(q))
        if q_max == 0:
            q_max = 1

        # determine exponentials for each action
        a_probs = np.zeros(self.agent.num_actions)
        for a in range(self.agent.num_actions):
            a_probs[a] = np.exp((q[a] / q_max) / self.temp)

        # normalize to get probabilities
        probs_sum = np.sum(a_probs)
        a_probs = np.true_divide(a_probs, probs_sum)

        # select randomly
        totals = np.cumsum(a_probs)
        norm = totals[-1]
        throw = self.rng.random_sample() * norm
        return np.searchsorted(totals, throw)

    def update(self, num_episode):
        pass


class ExpDecaySoftMax(SoftMaxExploration):
    """
    Represents a soft-max exploration strategy in which the temperature parameter decreases exponentially throughout
    training.
    """

    def __init__(self, max_episodes, max_temp=20, min_temp=0.1, rng=None):
        """
        Creates a new exploration strategy according to the given arguments.
        :param max_episodes: the maximum number of training episodes.
        :param float max_temp: the initial temperature, i.e., at the start of training.
        :param float min_temp: the final temperature, i.e., at the end of training.
        :param np.random.RandomState rng: the random number generator to be used by the selection strategy.
        """
        super().__init__(max_temp, rng)
        self.max_episodes = max_episodes
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.temp = 0.

    def update(self, num_episode):
        """
        Updates the temperature parameter of the exploration strategy as an exponential function of the episode number,
        i.e., where the temperature decreases exponentially with the number of training episodes.
        :param int num_episode: the number of the current training episode.
        :return:
        """
        # decreases temperature over time
        self.temp = self.min_temp + self.max_temp * max(0., (1 - (10 / self.max_episodes)) ** num_episode)
