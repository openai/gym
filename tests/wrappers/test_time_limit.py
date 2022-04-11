import numpy as np
import pytest

import gym
from gym.wrappers import TimeLimit


def test_time_limit_reset_info():
    env = gym.make("CartPole-v1")
    env = TimeLimit(env)
    ob_space = env.observation_space
    obs = env.reset()
    assert ob_space.contains(obs)
    del obs
    obs = env.reset(return_info=False)
    assert ob_space.contains(obs)
    del obs
    obs, info = env.reset(return_info=True)
    assert ob_space.contains(obs)
    assert isinstance(info, dict)
