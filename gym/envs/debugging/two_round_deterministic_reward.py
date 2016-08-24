"""
Simple environment with known optimal policy and value function.

Action 0 then 0 yields 0 reward and terminates the session.
Action 0 then 1 yields 3 reward and terminates the session.
Action 1 then 0 yields 1 reward and terminates the session.
Action 1 then 1 yields 2 reward and terminates the session.

Optimal policy: action 0 then 1.

Optimal value function v(observation): (this is a fully observable MDP so observation==state)

v(0)= 3 (you get observation 0 after taking action 0)
v(1)= 2 (you get observation 1 after taking action 1)
v(2)= 3 (you get observation 2 in the starting state)
"""

import gym
import random
from gym import spaces

class TwoRoundDeterministicRewardEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(3)
        self._reset()

    def _step(self, action):
        rewards = [[0, 3], [1, 2]]

        assert self.action_space.contains(action)

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
            return 2
        else:
            return self.firstAction

    def _reset(self):
        self.firstAction = None
        return self._get_obs()
