import numpy as np

import gym
from gym import spaces


class RescaleAction(gym.ActionWrapper):
    r"""Rescale the continuous action space of the environment from a range of [a, b]. """
    def __init__(self, env, a, b):
        assert isinstance(env.action_space, spaces.Box), (
            "expected Box action space, got {}".format(type(env.action_space)))
        assert np.less_equal(a, b).all(), (a, b)
        super(RescaleAction, self).__init__(env)
        self.a = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + a
            self.a = np.full(env.action_space.shape, a, dtype=env.action_space.dtype)
        else:
            self.a = a
        self.b = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + b
            self.b = np.full(env.action_space.shape, b, dtype=env.action_space.dtype)
        else:
            self.b = b

    def action(self, action):
        assert np.greater_equal(action, self.a).all() and np.less_equal(action, self.b).all()
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low)*((action - self.a)/(self.b - self.a))
        action = np.clip(action, low, high)
        return action
