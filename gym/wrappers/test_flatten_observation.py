import pytest

import gym
from gym.wrappers import FlattenObservation
try:
    import atari_py
except ImportError:
    atari_py = None


@pytest.mark.skipif(atari_py is None, reason='Only run this test when atari_py is installed')
@pytest.mark.parametrize('env_id', ['Pong-v0', 'SpaceInvaders-v0'])
def test_flatten_observation(env_id):
    env = gym.make(env_id)
    wrapped_env = FlattenObservation(env)

    obs = env.reset()
    wrapped_obs = wrapped_env.reset()

    assert len(obs.shape) == 3
    assert len(wrapped_obs.shape) == 1
    assert wrapped_obs.shape[0] == obs.shape[0]*obs.shape[1]*obs.shape[2]
