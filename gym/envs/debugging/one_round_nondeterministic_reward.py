import gym
import random
from gym import spaces

class OneRoundNondeterministicRewardEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(1)
        self._reset()

    def _step(self, action):
        assert(self.action_space.contains(action))
        if action:
            #your agent should figure out that this option has expected value 2.5
            reward = random.choice([0, 5])
        else:
            #your agent should figure out that this option has expected value 2.0
            reward = random.choice([1, 3])

        done = True
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return 0

    def _reset(self):
        return self._get_obs()
