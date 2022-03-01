"""
A class for providing an automatic reset functionality for gym environments when calling step.

If done was not true for the previous call to step, step returns

obs, reward, done, info

as normal.

If done was true for the previous call to step, the action is ignored, and self.env.reset is called, and step returns

obs, None, None, info

Warning: When using this wrapper to collect rollouts, note that the action given to .step() which causes the reset (first call to 
.step() after done = True) will have no effect on any observation, and should likely be excluded from logging.

"""


from typing import Optional

import numpy as np
import gym


class AutoResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._env_done = False

    def reset(self, **kwargs):
        self._env_done = False
        return self.env.reset(**kwargs)

    def step(self, action):
        if self._env_done:
            obs, info = self.reset(
                return_info=True
            )  # we are assuming the return_info behavior is implemented in environments using this wrapper
            return obs, None, None, info

        obs, reward, done, info = self.env.step(action)
        self._env_done = done

        return obs, reward, done, info
