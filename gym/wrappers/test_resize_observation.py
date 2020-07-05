import pytest

import gym
from gym.wrappers import ResizeObservation
try:
    import atari_py
except ImportError:
    atari_py = None


@pytest.mark.skipif(atari_py is None, reason='Only run this test when atari_py is installed')
@pytest.mark.parametrize('env_id', ['PongNoFrameskip-v0', 'SpaceInvadersNoFrameskip-v0'])
@pytest.mark.parametrize('shape', [16, 32, (8, 5), [10, 7]])
def test_resize_observation(env_id, shape):
    env = gym.make(env_id)
    env = ResizeObservation(env, shape)


    assert env.observation_space.shape[-1] == 3
    obs = env.reset()
    if isinstance(shape, int):
        assert env.observation_space.shape[:2] == (shape, shape)
        assert obs.shape == (shape, shape, 3)
    else:
        assert env.observation_space.shape[:2] == tuple(shape)
        assert obs.shape == tuple(shape) + (3,)
