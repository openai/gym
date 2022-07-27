import operator as op
from functools import reduce

import pytest

import gym
from tests.dev_wrappers.mock_data import (
    DICT_SPACE,
    DISCRETE_ACTION,
    FLATTENEND_DICT_SIZE,
)
from tests.dev_wrappers.utils import TestingEnv

try:
    from gym.wrappers import FlattenObservationsV0
except ImportError:
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize(
    ("env"),
    [
        gym.make(
            "CarRacing-v2",
            continuous=False,
            disable_env_checker=True,
            new_step_api=True,
        ),
    ],
)
def test_flatten_observation_v0(env):
    """Test correct flattening of observation space."""
    flattened_shape = reduce(op.mul, env.observation_space.shape, 1)
    wrapped_env = FlattenObservationsV0(env)
    wrapped_env.reset()

    obs, _, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    assert wrapped_env.observation_space.shape[0] == flattened_shape  # pyright: ignore
    assert obs.shape[0] == flattened_shape


@pytest.mark.parametrize(
    ("env", "flattened_size"),
    [(TestingEnv(observation_space=DICT_SPACE), FLATTENEND_DICT_SIZE)],
)
def test_dict_flatten_observation_v0(env, flattened_size):
    wrapped_env = FlattenObservationsV0(env)
    obs, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    assert wrapped_env.observation_space.shape[0] == flattened_size  # pyright: ignore
    assert obs.shape[0] == flattened_size
