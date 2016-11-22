from gym.utils import seeding, colorize
from gym import core, spaces

from six import StringIO
import sys
import numpy as np

BIG = 100000000


class RandomTabularMDPEnv(core.Env):
    """
    A tabular MDP with a fixed state / action space, where the rewards and transition probabilities are initialized
    randomly.
    The rewards follow a normal distribution with unit variance. The mean parameters themselves are initialized from
    i.i.d. Normal(1, 1).
    The transition probabilities are initialized from a flat Dirichlet prior, which put equal probability mass over
    the entire probability simplex.
    These distributions are commonly used as the prior distribution in Bayesian methods. Hence, they give such
    Bayesian methods the maximal advantage.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, n_states, n_actions, n_episodes, episode_length):
        """
        n_states: Number of states.
        n_actions: Number of actions.
        n_episodes: Number of episodes allowed to interact with a given MDP.
        episode_length: Length of each episode for the sampled MDP.
        """

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_episodes = n_episodes
        self.episode_length = episode_length

        self.observation_space = spaces.Tuple([
            # Current state
            spaces.Discrete(n_states),
            # Previous action
            spaces.Discrete(n_actions),
            # Previous reward
            spaces.Box(low=-BIG, high=BIG, shape=(1,)),
            # Previous termination flag
            spaces.Box(low=0, high=1, shape=(1,)),
        ])
        self.action_space = spaces.Discrete(n_actions)

        # Transition probability matrix
        self._P = None
        # Reward parameter
        self._R = None
        # Mean parameter for each arm's reward, which follows a Bernoulli distribution
        self._episode_cnt = None
        self._episode_t = None
        self._last_action = None
        self._last_reward = None
        self._last_terminal = None
        self._total_reward = None
        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self._R = self.np_random.normal(loc=1, scale=1, size=(self.n_states, self.n_actions))
        self._P = self.np_random.dirichlet(alpha=np.ones(self.n_states), size=(self.n_states, self.n_actions))
        self._episode_cnt = 0
        self._episode_t = 0
        self._current_state = 0
        # default value
        self._last_action = 0
        self._last_reward = 0
        self._last_terminal = 1
        self._total_reward = 0
        return self._get_obs()

    def _get_obs(self):
        return self._current_state, self._last_action, self._last_reward, self._last_terminal

    def _step(self, action):
        assert self.action_space.contains(action)
        p = self._P[self._current_state, action]
        next_state = np.where(self.np_random.multinomial(n=1, pvals=p))[0][0]
        reward = np.random.normal(loc=self._R[self._current_state, action], scale=1.)
        self._last_action = action
        self._last_reward = reward
        self._last_terminal = 0
        self._episode_t += 1
        self._current_state = next_state
        if self._episode_t >= self.episode_length:
            # reset episode
            self._episode_t = 0
            self._episode_cnt += 1
            self._current_state = 0
            self._last_terminal = 1
        self._total_reward += reward
        done = self._episode_cnt >= self.n_episodes
        obs = self._get_obs()
        return obs, reward, done, {}

    def _render(self, mode='human', close=False):
        """
        Renders the bandit environment in ASCII style. Closely resembles the rendering implementation of algorithmic
        tasks.
        """
        if close:
            # Nothing interesting to close
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('#Episode: {}\n'.format(self._episode_cnt))
        outfile.write('T within episode: {}\n'.format(self._episode_t))
        outfile.write('Total reward so far: {}\n'.format(self._total_reward))
        outfile.write('Current State: {}\n'.format(self._current_state))
        outfile.write('Action: {}\n'.format(self._last_action))

        return outfile
