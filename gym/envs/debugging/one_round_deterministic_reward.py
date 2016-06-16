"""
Simple environment with known optimal policy and value function.

This environment has just two actions.
Action 0 yields 0 reward and then terminates the session.
Action 1 yields 1 reward and then terminates the session.

Optimal policy: action 1.

Optimal value function: v(0)=1 (there is only one state, state 0)
"""

import gym
import random
from gym import spaces

class OneRoundDeterministicRewardEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(1)
        self._reset()

    def _step(self, action):
        assert self.action_space.contains(action)
        if action:
                reward = 1
        else:
                reward = 0

        done = True
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return 0

    def _reset(self):
        return self._get_obs()
