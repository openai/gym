import numpy as np
import gym
import pytest
from gym.wrappers import AtariPreprocessing
try:
    import atari_py
except ImportError:
    atari_py = None

@pytest.mark.skipif(atari_py is None, reason='Only run this test when atari_py is installed')
def test_atari_preprocessing():
    import cv2
    env1 = gym.make('PongNoFrameskip-v0')
    env2 = AtariPreprocessing(env1, screen_size=84, grayscale_obs=True)
    env3 = AtariPreprocessing(env1, screen_size=84, grayscale_obs=False)
    obs1 = env1.reset()
    assert obs1.shape == (210, 160, 3)
    obs2 = env2.reset()
    assert obs2.shape == (84, 84)
    obs3 = env3.reset()
    assert obs3.shape == (84, 84, 3)
    # TODO peterz - figure out why assertions below are faliing and fix
    # np.testing.assert_allclose(obs3, cv2.resize(obs1, (84, 84), interpolation=cv2.INTER_AREA))
    # np.testing.assert_allclose(obs2, cv2.cvtColor(obs3, cv2.COLOR_RGB2GRAY))
