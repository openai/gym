import pytest

import gym
from gym import spaces
from gym.wrappers import ResizeObservation


@pytest.mark.parametrize("env_id", ["CarRacing-v2"])
@pytest.mark.parametrize("shape", [16, 32, (8, 5), [10, 7]])
def test_resize_observation(env_id, shape):
    env = gym.make(env_id, disable_env_checker=True)
    env = ResizeObservation(env, shape)

    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape[-1] == 3
    obs = env.reset()
    if isinstance(shape, int):
        assert env.observation_space.shape[:2] == (shape, shape)
        assert obs.shape == (shape, shape, 3)
    else:
        assert env.observation_space.shape[:2] == tuple(shape)
        assert obs.shape == tuple(shape) + (3,)
