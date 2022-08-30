import numpy as np
import pytest

import gym
from gym.wrappers import TransformObservation


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
def test_transform_observation(env_id):
    def affine_transform(x):
        return 3 * x + 2

    env = gym.make(env_id, disable_env_checker=True)
    wrapped_env = TransformObservation(
        gym.make(env_id, disable_env_checker=True), lambda obs: affine_transform(obs)
    )

    obs, info = env.reset(seed=0)
    wrapped_obs, wrapped_obs_info = wrapped_env.reset(seed=0)
    assert np.allclose(wrapped_obs, affine_transform(obs))
    assert isinstance(wrapped_obs_info, dict)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    (
        wrapped_obs,
        wrapped_reward,
        wrapped_terminated,
        wrapped_truncated,
        _,
    ) = wrapped_env.step(action)
    assert np.allclose(wrapped_obs, affine_transform(obs))
    assert np.allclose(wrapped_reward, reward)
    assert wrapped_terminated == terminated
    assert wrapped_truncated == truncated
