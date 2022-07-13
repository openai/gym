"""Tests for the filter observation wrapper."""
from typing import Optional

import numpy as np
import pytest

import gym
from gym.spaces import Box, Dict, Tuple
from gym.wrappers import FilterObservation, FlattenObservation


class FakeEnvironment(gym.Env):
    def __init__(self, observation_space, render_mode=None):
        self.observation_space = observation_space
        self.obs_keys = self.observation_space.spaces.keys()
        self.action_space = Box(shape=(1,), low=-1, high=1, dtype=np.float32)
        self.render_mode = render_mode

    def render(self, mode="human"):
        image_shape = (32, 32, 3)
        return np.zeros(image_shape, dtype=np.uint8)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        observation = self.observation_space.sample()
        return observation, {}

    def step(self, action):
        del action
        observation = self.observation_space.sample()
        reward, terminal, info = 0.0, False, {}
        return observation, reward, terminal, info


NESTED_DICT_TEST_CASES = (
    (
        Dict(
            {
                "key1": Box(shape=(2,), low=-1, high=1, dtype=np.float32),
                "key2": Dict(
                    {
                        "subkey1": Box(shape=(2,), low=-1, high=1, dtype=np.float32),
                        "subkey2": Box(shape=(2,), low=-1, high=1, dtype=np.float32),
                    }
                ),
            }
        ),
        (6,),
    ),
    (
        Dict(
            {
                "key1": Box(shape=(2, 3), low=-1, high=1, dtype=np.float32),
                "key2": Box(shape=(), low=-1, high=1, dtype=np.float32),
                "key3": Box(shape=(2,), low=-1, high=1, dtype=np.float32),
            }
        ),
        (9,),
    ),
    (
        Dict(
            {
                "key1": Tuple(
                    (
                        Box(shape=(2,), low=-1, high=1, dtype=np.float32),
                        Box(shape=(2,), low=-1, high=1, dtype=np.float32),
                    )
                ),
                "key2": Box(shape=(), low=-1, high=1, dtype=np.float32),
                "key3": Box(shape=(2,), low=-1, high=1, dtype=np.float32),
            }
        ),
        (7,),
    ),
    (
        Dict(
            {
                "key1": Tuple((Box(shape=(2,), low=-1, high=1, dtype=np.float32),)),
                "key2": Box(shape=(), low=-1, high=1, dtype=np.float32),
                "key3": Box(shape=(2,), low=-1, high=1, dtype=np.float32),
            }
        ),
        (5,),
    ),
    (
        Dict(
            {
                "key1": Tuple(
                    (Dict({"key9": Box(shape=(2,), low=-1, high=1, dtype=np.float32)}),)
                ),
                "key2": Box(shape=(), low=-1, high=1, dtype=np.float32),
                "key3": Box(shape=(2,), low=-1, high=1, dtype=np.float32),
            }
        ),
        (5,),
    ),
)


class TestNestedDictWrapper:
    @pytest.mark.parametrize("observation_space, flat_shape", NESTED_DICT_TEST_CASES)
    def test_nested_dicts_size(self, observation_space, flat_shape):
        env = FakeEnvironment(observation_space=observation_space)

        # Make sure we are testing the right environment for the test.
        observation_space = env.observation_space
        assert isinstance(observation_space, Dict)

        wrapped_env = FlattenObservation(FilterObservation(env, env.obs_keys))
        assert wrapped_env.observation_space.shape == flat_shape

        assert wrapped_env.observation_space.dtype == np.float32

    @pytest.mark.parametrize("observation_space, flat_shape", NESTED_DICT_TEST_CASES)
    def test_nested_dicts_ravel(self, observation_space, flat_shape):
        env = FakeEnvironment(observation_space=observation_space)
        wrapped_env = FlattenObservation(FilterObservation(env, env.obs_keys))
        obs, info = wrapped_env.reset()
        assert obs.shape == wrapped_env.observation_space.shape
        assert isinstance(info, dict)
