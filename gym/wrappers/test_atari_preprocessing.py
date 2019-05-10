import cv2
import numpy as np

import gym
from gym.wrappers import AtariPreprocessing


def test_atari_preprocessing():
    env1 = gym.make('PongNoFrameskip-v0')
    env2 = AtariPreprocessing(env1, screen_size=84, grayscale_obs=True)
    env3 = AtariPreprocessing(env1, screen_size=84, grayscale_obs=False)
    obs1 = env1.reset()
    assert obs1.shape == (210, 160, 3)
    obs2 = env2.reset()
    assert obs2.shape == (84, 84)
    obs3 = env3.reset()
    assert obs3.shape == (84, 84, 3)
    assert np.allclose(obs3, cv2.resize(obs1, (84, 84), interpolation=cv2.INTER_AREA))
    assert np.allclose(obs2, cv2.cvtColor(obs3, cv2.COLOR_RGB2GRAY))
