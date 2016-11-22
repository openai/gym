from gym.utils import seeding, colorize
from gym import core, spaces

from six import StringIO
import sys

BIG = 100000000


class BernoulliBanditEnv(core.Env):
    """
    An implementation of the classical multi-armed bandit task. Each arm has a Bernoulli reward distribution,
    where the mean parameter is sampled from Uniform(0, 1).
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, n_arms, n_episodes):
        """
        n_arms: Number of arms.
        n_episodes: Number of episodes. For bandits, each episode only lasts one time step. So this is simply the
        number of pulls allowed.
        """

        self.n_arms = n_arms
        self.n_episodes = n_episodes
        self.observation_space = spaces.Tuple([
            # State: just a placeholder for bandits
            spaces.Discrete(1),
            # Previous action
            spaces.Discrete(n_arms),
            # Previous reward
            spaces.Box(low=-BIG, high=BIG, shape=(1,)),
            # Previous termination flag
            # Since a bandit task always terminates after one time step, the termination flag is always on
            spaces.Box(low=0, high=1, shape=(1,)),
        ])
        self.action_space = spaces.Discrete(n_arms)
        # Mean parameter for each arm's reward, which follows a Bernoulli distribution
        self._arm_means = None
        self._episode_cnt = None
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
        self._arm_means = self.np_random.uniform(low=0, high=1, size=(self.n_arms,))
        self._episode_cnt = 0
        # Default value
        self._last_action = 0
        self._last_reward = 0
        self._last_terminal = 1
        self._total_reward = 0
        return self._get_obs()

    def _get_obs(self):
        return 0, self._last_action, self._last_reward, self._last_terminal

    def _step(self, action):
        assert self.action_space.contains(action)
        reward = self.np_random.binomial(n=1, p=self._arm_means[action])
        self._last_action = action
        self._last_reward = reward
        self._last_terminal = 1
        self._episode_cnt += 1
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
        outfile.write('Total reward so far: {}\n'.format(self._total_reward))
        outfile.write('Action:')
        for idx, mean in enumerate(self._arm_means):
            if idx == self._last_action:
                if self._last_reward == 1:
                    outfile.write(' ' + colorize('%.2f' % mean, 'green', highlight=True))
                else:
                    outfile.write(' ' + colorize('%.2f' % mean, 'red', highlight=True))
            else:
                outfile.write(' ' + '%.2f' % mean)
        outfile.write('\n')

        return outfile
