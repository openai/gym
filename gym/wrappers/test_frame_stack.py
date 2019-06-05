import pytest

import numpy as np

import gym
from gym.wrappers import FrameStack
try:
    import atari_py
except ImportError:
    atari_py = None
try:
    import lz4
except ImportError:
    lz4 = None


@pytest.mark.skipif(atari_py is None, reason='Only run this test when atari_py is installed')
@pytest.mark.skipif(lz4 is None, reason='Only run this test when lz4 is installed')
@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0', 'Pong-v0'])
@pytest.mark.parametrize('num_stack', [2, 3, 4])
@pytest.mark.parametrize('lz4_compress', [True, False])
def test_frame_stack(env_id, num_stack, lz4_compress):
    env = gym.make(env_id)
    shape = env.observation_space.shape
    env = FrameStack(env, num_stack, lz4_compress)
    assert env.observation_space.shape == (num_stack,) + shape

    obs = env.reset()
    obs = np.asarray(obs)
    assert obs.shape == (num_stack,) + shape
    for i in range(1, num_stack):
        assert np.allclose(obs[i - 1], obs[i])

    obs, _, _, _ = env.step(env.action_space.sample())
    obs = np.asarray(obs)
    assert obs.shape == (num_stack,) + shape
    for i in range(1, num_stack - 1):
        assert np.allclose(obs[i - 1], obs[i])
    assert not np.allclose(obs[-1], obs[-2])
