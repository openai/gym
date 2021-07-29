import numpy as np
import warnings
from gym import ActionWrapper
from gym.spaces import Box


class ClipAction(ActionWrapper):
    r"""Clip the continuous action within the valid bound."""

    def __init__(self, env):
        assert isinstance(env.action_space, Box)
        warnings.warn("Gym\'s internal preprocessing wrappers are now deprecated. While they will continue to work for the foreseeable future, we strongly recommend using SuperSuit instead: https://github.com/PettingZoo-Team/SuperSuit")
        super(ClipAction, self).__init__(env)

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)
