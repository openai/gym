import numpy as np
import gym
from gym.wrappers import AtariPreprocessing
import pytest

pytest.importorskip('atari_py')


@pytest.fixture(scope='module')
def env_fn():
    return lambda: gym.make('PongNoFrameskip-v4')


def test_atari_preprocessing_grayscale(env_fn):
    import cv2
    env1 = env_fn()
    env2 = AtariPreprocessing(env_fn(), screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=0)
    env3 = AtariPreprocessing(env_fn(), screen_size=84, grayscale_obs=False, frame_skip=1, noop_max=0)
    env1.reset()
    # take these steps to imitate actions of FireReset logic
    env1.step(1)
    obs1 = env1.step(2)[0]
    obs2 = env2.reset()
    obs3 = env3.reset()
    assert obs1.shape == (210, 160, 3)
    assert obs2.shape == (84, 84)
    assert obs3.shape == (84, 84, 3)
    np.testing.assert_allclose(obs3, cv2.resize(obs1, (84, 84), interpolation=cv2.INTER_AREA))
    obs3_gray = cv2.cvtColor(obs3, cv2.COLOR_RGB2GRAY)
    # the edges of the numbers do not render quite the same in the grayscale, so we ignore them
    np.testing.assert_allclose(obs2[10:], obs3_gray[10:])

    env1.close()
    env2.close()
    env3.close()


def test_atari_preprocessing_scale(env_fn):
    for grayscale in [False, True]:
        env_scale = AtariPreprocessing(env_fn(), screen_size=84, grayscale_obs=grayscale, scale_obs=True,
                                       frame_skip=1, noop_max=0)
        env_no_scale = AtariPreprocessing(env_fn(), screen_size=84, grayscale_obs=grayscale, scale_obs=False,
                                          frame_skip=1, noop_max=0)

        # arbitrary chosen number for stepping into environment and ensuring all observations are in the required range
        test_steps = 10

        obs_scale = env_scale.reset().flatten()
        for steps in range(test_steps):
            obs, _, done, _ = env_scale.step(env_scale.action_space.sample())
            assert (0 <= obs_scale).all() and (obs_scale <= 1).all(), 'All values must be in range [0,1]'
            if done:
                break

        obs_no_scale = env_no_scale.reset().flatten()
        for steps in range(test_steps):
            obs, _, done, _ = env_scale.step(env_scale.action_space.sample())
            assert (0 <= obs_no_scale).all() and (obs_no_scale <= 255).all(), 'All values must be in range [0,255]'
            if done:
                break

        env_scale.close()
        env_no_scale.close()
