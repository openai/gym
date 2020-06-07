"""
Implements BitFlip env from Andrychowicz et al. '17
(https://arxiv.org/abs/1707.01495)
"""

import gym
import numpy as np

from gym import spaces
from gym.utils import seeding


class BitFlip(gym.GoalEnv):
    """
    Simple BitFlipping environment for goal-conditioned reinforcement
    learning.

    BitFlipEnv is a Goal-conditioned environment where the agent must reach
    a goal state of flipped bits from the current state, only receiving a
    sparse reward of -1 or 0 depending on if the goal state is reached.

    The environment follows the toy motivating example presented in
    Andrychowicz et al. '17, where at each step, the agent can choose
    whether which bit in the string it will flip the value for. The episode
    terminates when either the goal state has been reached, or num_bits steps
    have been taken.

    An alternate version of this environment is a 2D action space where the
    agent can only flip the ith bit, where i is the step count in the episode.

    See Section 3.1 from Andrychowicz et al. for the full description.
    """

    def __init__(self, num_bits=10, select_bit=True, succ_bonus=11):
        """
        Initializes Bit Flipping environment instance
        Args:
          num_bits: length of 1D binary string for state and goal
          flip_mode: whether or not agent can choose which bit to flip
        """
        self.num_bits = num_bits
        self.select_bit = select_bit
        self.succ_bonus = succ_bonus
        self.observation_space = spaces.Dict({
            "observation": spaces.MultiBinary(num_bits),
            "desired_goal": spaces.MultiBinary(num_bits),
            "achieved_goal": spaces.MultiBinary(num_bits)})
        if select_bit:
            self.action_space = spaces.Discrete(num_bits)
        else:
            self.action_space = spaces.Discrete(2)
        self.state, self.goal, self.current_step = None, None, None
        self.done = False
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.done = False
        self.current_step = 0
        self.state = self.np_random.randint(2, size=self.num_bits)
        self.goal = self.np_random.randint(2, size=self.num_bits)
        return {'observation': self.state,
                'desired_goal': self.goal,
                'achieved_goal': self.state}

    def _is_success(self, achieved_goal, desired_goal):
        return np.all(achieved_goal == desired_goal)

    def step(self, action):
        assert action in self.action_space, 'action {} invalid'.format(action)
        if self.done:
            print('Episode has ended, need to call reset()')
        else:
            select_idx = None
            if self.select_bit:
                select_idx = action
            elif action:
                select_idx = self.current_step
            if select_idx is not None:
                self.state[select_idx] = int(not self.state[select_idx])
            self.current_step += 1
        obs = {'observation': self.state.copy(),
               'desired_goal': self.goal.copy(),
               'achieved_goal': self.state.copy()}
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'current_step': self.current_step-1
        }
        reward = self.compute_reward(self.state, self.goal, info)
        self.done = done = (self.current_step == self.num_bits) or (reward == 0)
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        succ = self._is_success(achieved_goal, desired_goal)
        if not self.select_bit:
            succ += self.succ_bonus
        return succ - 1
