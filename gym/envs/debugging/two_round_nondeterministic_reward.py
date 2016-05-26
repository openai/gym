import gym
import random
from gym import spaces

class TwoRoundNondeterministicRewardEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(3)
        self._reset()

    def _step(self, action):
        rewards = [
            [
                [-1, 1], #expected value 0
                [0, 0, 9] #expected value 3. This is the best path.
            ],
            [
                [0, 2], #expected value 1
                [2, 3] #expected value 2
            ]
        ]

        assert(self.action_space.contains(action))

        if self.firstAction is None:
            self.firstAction = action
            reward = 0
            done = False
        else:
            reward = random.choice(rewards[self.firstAction][action])
            done = True

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        if self.firstAction is None:
            return 2
        else:
            return self.firstAction

    def _reset(self):
        self.firstAction = None
        return self._get_obs()
