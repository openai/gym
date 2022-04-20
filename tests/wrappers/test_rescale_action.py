import numpy as np
import pytest

import gym
from gym.wrappers import RescaleAction


def test_rescale_action():
    env = gym.make("CartPole-v1", return_two_dones=True)
    with pytest.raises(AssertionError):
        env = RescaleAction(env, -1, 1)
    del env

    env = gym.make("Pendulum-v1", return_two_dones=True)
    wrapped_env = RescaleAction(gym.make("Pendulum-v1", return_two_dones=True), -1, 1)

    seed = 0

    obs = env.reset(seed=seed)
    wrapped_obs = wrapped_env.reset(seed=seed)
    assert np.allclose(obs, wrapped_obs)

    obs, reward, _, _, _ = env.step([1.5])
    with pytest.raises(AssertionError):
        wrapped_env.step([1.5])
    wrapped_obs, wrapped_reward, _, _, _ = wrapped_env.step([0.75])

    assert np.allclose(obs, wrapped_obs)
    assert np.allclose(reward, wrapped_reward)
