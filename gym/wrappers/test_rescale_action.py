import pytest

import numpy as np

import gym
from gym.wrappers import RescaleAction


def test_rescale_action():
    env = gym.make('CartPole-v1')
    with pytest.raises(AssertionError):
        env = RescaleAction(env, -1, 1)
    del env

    env = gym.make('Pendulum-v0')
    wrapped_env = RescaleAction(gym.make('Pendulum-v0'), -1, 1)

    seed = 0
    env.seed(seed)
    wrapped_env.seed(seed)

    obs = env.reset()
    wrapped_obs = wrapped_env.reset()
    assert np.allclose(obs, wrapped_obs)

    obs, reward, _, _ = env.step([1.5])
    with pytest.raises(AssertionError):
        wrapped_env.step([1.5])
    wrapped_obs, wrapped_reward, _, _ = wrapped_env.step([0.75])

    assert np.allclose(obs, wrapped_obs)
    assert np.allclose(reward, wrapped_reward)
