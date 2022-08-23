"""Tests for the flatten observation wrapper."""

from collections import OrderedDict
from typing import Optional

import numpy as np
import pytest

import gym
from gym.spaces import Box, Dict, flatten, unflatten
from gym.wrappers import FlattenObservation


class FakeEnvironment(gym.Env):
    def __init__(self, observation_space):
        self.observation_space = observation_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.observation = self.observation_space.sample()
        return self.observation, {}


OBSERVATION_SPACES = (
    (
        Dict(
            OrderedDict(
                [
                    ("key1", Box(shape=(2, 3), low=0, high=0, dtype=np.float32)),
                    ("key2", Box(shape=(), low=1, high=1, dtype=np.float32)),
                    ("key3", Box(shape=(2,), low=2, high=2, dtype=np.float32)),
                ]
            )
        ),
        True,
    ),
    (
        Dict(
            OrderedDict(
                [
                    ("key2", Box(shape=(), low=0, high=0, dtype=np.float32)),
                    ("key3", Box(shape=(2,), low=1, high=1, dtype=np.float32)),
                    ("key1", Box(shape=(2, 3), low=2, high=2, dtype=np.float32)),
                ]
            )
        ),
        True,
    ),
    (
        Dict(
            {
                "key1": Box(shape=(2, 3), low=-1, high=1, dtype=np.float32),
                "key2": Box(shape=(), low=-1, high=1, dtype=np.float32),
                "key3": Box(shape=(2,), low=-1, high=1, dtype=np.float32),
            }
        ),
        False,
    ),
)


class TestFlattenEnvironment:
    @pytest.mark.parametrize("observation_space, ordered_values", OBSERVATION_SPACES)
    def test_flattened_environment(self, observation_space, ordered_values):
        """
        make sure that flattened observations occur in the order expected
        """
        env = FakeEnvironment(observation_space=observation_space)
        wrapped_env = FlattenObservation(env)
        flattened, info = wrapped_env.reset()

        unflattened = unflatten(env.observation_space, flattened)
        original = env.observation

        self._check_observations(original, flattened, unflattened, ordered_values)

    @pytest.mark.parametrize("observation_space, ordered_values", OBSERVATION_SPACES)
    def test_flatten_unflatten(self, observation_space, ordered_values):
        """
        test flatten and unflatten functions directly
        """
        original = observation_space.sample()

        flattened = flatten(observation_space, original)
        unflattened = unflatten(observation_space, flattened)

        self._check_observations(original, flattened, unflattened, ordered_values)

    def _check_observations(self, original, flattened, unflattened, ordered_values):
        # make sure that unflatten(flatten(original)) == original
        assert set(unflattened.keys()) == set(original.keys())
        for k, v in original.items():
            np.testing.assert_allclose(unflattened[k], v)

        if ordered_values:
            # make sure that the values were flattened in the order they appeared in the
            # OrderedDict
            np.testing.assert_allclose(sorted(flattened), flattened)
