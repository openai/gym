import pytest

import gym
from gym import spaces
from gym.wrappers import GrayScaleObservation


@pytest.mark.parametrize("env_id", ["CarRacing-v2"])
@pytest.mark.parametrize("keep_dim", [True, False])
def test_gray_scale_observation(env_id, keep_dim):
    rgb_env = gym.make(env_id, disable_env_checker=True)

    assert isinstance(rgb_env.observation_space, spaces.Box)
    assert len(rgb_env.observation_space.shape) == 3
    assert rgb_env.observation_space.shape[-1] == 3

    wrapped_env = GrayScaleObservation(rgb_env, keep_dim=keep_dim)
    assert isinstance(wrapped_env.observation_space, spaces.Box)
    if keep_dim:
        assert len(wrapped_env.observation_space.shape) == 3
        assert wrapped_env.observation_space.shape[-1] == 1
    else:
        assert len(wrapped_env.observation_space.shape) == 2

    wrapped_obs = wrapped_env.reset()
    assert wrapped_obs in wrapped_env.observation_space
