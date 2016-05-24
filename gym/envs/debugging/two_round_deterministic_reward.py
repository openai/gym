import gym
import random
from gym import spaces

class TwoRoundDeterministicRewardEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)
        self._reset()

    def _step(self, action):
        rewards = [[0, 3], [1, 2]]

        assert(self.action_space.contains(action))

        if self.firstAction is None:
            self.firstAction = action
            reward = 0
            done = False
        else:
            reward = rewards[self.firstAction][action]
            done = True

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        if self.firstAction is None:
            return 0
        else:
            return 1

    def _reset(self):
        self.firstAction = None
        return self._get_obs()
