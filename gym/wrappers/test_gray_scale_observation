import pytest

import gym
from gym.wrappers import GrayScaleObservation


@pytest.mark.parametrize('env_id', ['Pong-v0', 'SpaceInvaders-v0'])
@pytest.mark.parametrize('keep_dim', [True, False])
def test_gray_scale_observation(env_id, keep_dim):
    env = gym.make(env_id)
    wrapped_env = GrayScaleObservation(env, keep_dim=keep_dim)

    obs = env.reset()
    wrapped_obs = wrapped_env.reset()

    assert env.observation_space.shape[:2] == wrapped_env.observation_space.shape[:2]
    if keep_dim:
        assert wrapped_env.observation_space.shape[-1] == 1
        assert len(wrapped_obs.shape) == 3
    else:
        assert len(wrapped_env.observation_space.shape) == 2
        assert len(wrapped_obs.shape) == 2
