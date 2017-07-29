from pyfiction.simulators.games.machineofdeath_simulator import MachineOfDeathSimulator

import gym
from gym import spaces
from gym.utils import seeding


class MachineOfDeathEnv(gym.Env):
    """Machine of Death text game environment
    There are 3 game branches and more than 14 different endings with different rewards;
     the optimal cumulative rewards in the three game branches are close to 17.4, 18.5 and 28.5

    Available to play online at:
     http://ifarchive.giga.or.at/if-archive/games/competition2013/web/machineofdeath/MachineOfDeath.html

    The returned observation is a tuple (state, actions), where state is the complete state descriptions
     and actions is a list of action substrings sorted by their index (0..len(actions)-1).
    """

    def __init__(self):
        # number of actions is variable:
        self.action_space = spaces.Discrete(0)

        # observation space is unbounded:
        self.observation_space = spaces.Discrete(0)

        self.reward_range = (-30, 30)

        self._seed()
        self.simulator = MachineOfDeathSimulator()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)

        self.simulator.write(action)
        state, actions, reward = self.simulator.read()

        self.action_space = spaces.Discrete(len(actions))

        return (state, actions), reward, bool(actions), {}

    def _reset(self):
        self.simulator.restart()

        state, actions, _ = self.simulator.read()

        self.action_space = spaces.Discrete(len(actions))

        return state, actions
