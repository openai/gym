from pyfiction.simulators.games.savingjohn_simulator import SavingJohnSimulator
# Import this when using this env (!): from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode

import gym
from gym import spaces
from gym.utils import seeding


class SavingJohnEnv(gym.Env):
    """Saving John text game environment
    There are 5 different endings with different rewards;
     the optimal cumulative reward is 19.4.

    Available to play online at:
     http://www.ifarchive.org/if-archive/games/competition2013/web/savingjohn/Saving%20John.html

    The returned observation is a tuple (state, actions), where state is the complete state descriptions
     and actions is a list of action substrings sorted by their index (0..len(actions)-1).
    """

    def __init__(self):
        # number of actions is variable:
        self.action_space = spaces.Discrete(0)

        # observation space is unbounded:
        self.observation_space = spaces.Discrete(0)

        self.reward_range = (-20, 20)

        self._seed()

        try:
            self.simulator = SavingJohnSimulator()
        except AttributeError as e:
            print('AttributeError: ' + str(e) + '\nTo fix the StoryNode attribute error, please import:\n' +
                  '  from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode\n' +
                  'when calling the Saving John simulator.')

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
