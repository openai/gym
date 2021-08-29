import pytest

import numpy as np

import gym
from gym.wrappers import FlattenObservation
from gym import spaces


@pytest.mark.parametrize("env_id", ["Blackjack-v0", "KellyCoinflip-v0"])
def test_flatten_observation(env_id):
    env = gym.make(env_id)
    wrapped_env = FlattenObservation(env)

    obs = env.reset()
    wrapped_obs = wrapped_env.reset()

    if env_id == "Blackjack-v0":
        space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))
        )
        wrapped_space = spaces.Box(0, 1, [32 + 11 + 2], dtype=np.int64)
    elif env_id == "KellyCoinflip-v0":
        space = spaces.Tuple(
            (spaces.Box(0, 250.0, [1], dtype=np.float32), spaces.Discrete(300 + 1))
        )
        low = np.zeros((302,), dtype=np.float64)
        high = np.array([250.0] + [1.0] * 301, dtype=np.float64)
        wrapped_space = spaces.Box(low, high, [1 + (300 + 1)], dtype=np.float64)

    assert space.contains(obs)
    assert wrapped_space.contains(wrapped_obs)
