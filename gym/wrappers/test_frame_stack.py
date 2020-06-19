import pytest
pytest.importorskip("atari_py")

import numpy as np
import gym
from gym.wrappers import FrameStack
try:
    import lz4
except ImportError:
    lz4 = None


@pytest.mark.parametrize('env_id', ['CartPole-v1', 'Pendulum-v0', 'Pong-v0'])
@pytest.mark.parametrize('num_stack', [2, 3, 4])
@pytest.mark.parametrize('lz4_compress', [
    pytest.param(True, marks=pytest.mark.skipif(lz4 is None, reason="Need lz4 to run tests with compression")),
    False
])
def test_frame_stack(env_id, num_stack, lz4_compress):
    env = gym.make(env_id)
    shape = env.observation_space.shape
    env = FrameStack(env, num_stack, lz4_compress)
    assert env.observation_space.shape == (num_stack,) + shape
    assert env.observation_space.dtype == env.env.observation_space.dtype

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

    obs, _, _, _ = env.step(env.action_space.sample())
    assert len(obs) == num_stack
