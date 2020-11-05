import pytest

import numpy as np

import gym
from gym.wrappers import GrayScaleObservation
from gym.wrappers import AtariPreprocessing
pytest.importorskip('atari_py')
pytest.importorskip('cv2')

@pytest.mark.parametrize('env_id', ['PongNoFrameskip-v0', 'SpaceInvadersNoFrameskip-v0'])
@pytest.mark.parametrize('keep_dim', [True, False])
def test_gray_scale_observation(env_id, keep_dim):
    gray_env = AtariPreprocessing(gym.make(env_id), screen_size=84, grayscale_obs=True)
    rgb_env = AtariPreprocessing(gym.make(env_id), screen_size=84, grayscale_obs=False)
    wrapped_env = GrayScaleObservation(rgb_env, keep_dim=keep_dim)
    assert rgb_env.observation_space.shape[-1] == 3

    seed = 0
    gray_env.seed(seed)
    wrapped_env.seed(seed)

    gray_obs = gray_env.reset()
    wrapped_obs = wrapped_env.reset()

    if keep_dim:
        assert wrapped_env.observation_space.shape[-1] == 1
        assert len(wrapped_obs.shape) == 3
        wrapped_obs = wrapped_obs.squeeze(-1)
    else:
        assert len(wrapped_env.observation_space.shape) == 2
        assert len(wrapped_obs.shape) == 2

    # ALE gray scale is slightly different, but no more than by one shade
    assert np.allclose(gray_obs.astype('int32'), wrapped_obs.astype('int32'), atol=1)
