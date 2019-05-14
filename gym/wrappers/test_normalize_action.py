import pytest

import gym
from gym.wrappers import NormalizeAction


def test_normalize_action():
    env = gym.make('CartPole-v1')
    with pytest.raises(AssertionError):
        env = NormalizeAction(env)
    del env

    env = gym.make('Pendulum-v0')
    env = NormalizeAction(env)
    env.reset()
    with pytest.raises(AssertionError):
        env.step(10+env.action_space.sample())
