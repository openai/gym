import numpy as np
import pytest

import gym
from gym.wrappers import RescaleAction


def test_rescale_action():
    env = gym.make("CartPole-v1")
    with pytest.raises(AssertionError):
        env = RescaleAction(env, -1, 1)
    del env

    env = gym.make("Pendulum-v1")
    wrapped_env = RescaleAction(gym.make("Pendulum-v1"), -1, 1)

    seed = 0

    obs = env.reset(seed=seed)
    wrapped_obs = wrapped_env.reset(seed=seed)
    assert np.allclose(obs, wrapped_obs)

    obs, reward, _, _ = env.step(np.array([1.5], dtype=env.action_space.dtype))
    with pytest.raises(AssertionError):
        wrapped_env.step(np.array([1.5], dtype=env.action_space.dtype))
    wrapped_obs, wrapped_reward, _, _ = wrapped_env.step(
        np.array([0.75], dtype=env.action_space.dtype)
    )

    assert np.allclose(obs, wrapped_obs)
    assert np.allclose(reward, wrapped_reward)
