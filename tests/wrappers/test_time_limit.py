import pytest

import numpy as np

import gym
from gym.wrappers import TimeLimit


def test_time_limit_reset_info():
    env = gym.make("CartPole-v1")
    env = TimeLimit(env)
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    del obs
    obs = env.reset(return_info=False)
    assert isinstance(obs, np.ndarray)
    del obs
    obs, info = env.reset(return_info=True)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
