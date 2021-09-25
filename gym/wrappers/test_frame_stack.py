import pytest

pytest.importorskip("gym.envs.atari")

import numpy as np
import gym
from gym.wrappers import FrameStack

try:
    import lz4
except ImportError:
    lz4 = None


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1", "Pong-v0"])
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
    env = gym.make(env_id)
    env.seed(0)
    shape = env.observation_space.shape
    env = FrameStack(env, num_stack, lz4_compress)
    assert env.observation_space.shape == (num_stack,) + shape
    assert env.observation_space.dtype == env.env.observation_space.dtype

    dup = gym.make(env_id)
    dup.seed(0)

    obs = env.reset()
    dup_obs = dup.reset()
    assert np.allclose(obs[-1], dup_obs)

    for _ in range(num_stack ** 2):
        action = env.action_space.sample()
        dup_obs, _, _, _ = dup.step(action)
        obs, _, _, _ = env.step(action)
        assert np.allclose(obs[-1], dup_obs)

    assert len(obs) == num_stack
