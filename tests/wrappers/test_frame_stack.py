import numpy as np
import pytest

import gym
from gym.wrappers import FrameStack

try:
    import lz4
except ImportError:
    lz4 = None


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1", "CarRacing-v2"])
@pytest.mark.parametrize("num_stack", [2, 3, 4])
@pytest.mark.parametrize(
    "lz4_compress",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                lz4 is None, reason="Need lz4 to run tests with compression"
            ),
        ),
        False,
    ],
)
def test_frame_stack(env_id, num_stack, lz4_compress):
    env = gym.make(env_id, disable_env_checker=True)
    shape = env.observation_space.shape
    env = FrameStack(env, num_stack, lz4_compress)
    assert env.observation_space.shape == (num_stack,) + shape
    assert env.observation_space.dtype == env.env.observation_space.dtype

    dup = gym.make(env_id, disable_env_checker=True)

    obs = env.reset(seed=0)
    dup_obs = dup.reset(seed=0)
    assert np.allclose(obs[-1], dup_obs)

    for _ in range(num_stack**2):
        action = env.action_space.sample()
        dup_obs, _, dup_done, _ = dup.step(action)
        obs, _, done, _ = env.step(action)

        assert dup_done == done
        assert np.allclose(obs[-1], dup_obs)

        if done:
            break

    assert len(obs) == num_stack
