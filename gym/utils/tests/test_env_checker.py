import gym
import numpy as np
import pytest
from gym.spaces import Box, Dict, Discrete

from gym.utils.env_checker import check_env


class ActionDictTestEnv(gym.Env):
    action_space = Dict({"position": Discrete(1), "velocity": Discrete(1)})
    observation_space = Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)

    def step(self, action):
        observation = np.array([1.0, 1.5, 0.5])
        reward = 1
        done = True
        info = {}
        return observation, reward, done, info

    def reset(self):
        return np.array([1.0, 1.5, 0.5])

    def render(self, mode="human"):
        pass


def test_check_env_dict_action():
    test_env = ActionDictTestEnv()

    with pytest.warns(Warning):
        check_env(env=test_env, warn=True)
