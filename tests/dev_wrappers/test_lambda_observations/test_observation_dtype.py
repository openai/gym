from collections import OrderedDict

import numpy as np
import pytest

import gym
from tests.dev_wrappers.mock_data import (
    DICT_SPACE,
    DISCRETE_ACTION,
    DOUBLY_NESTED_DICT_SPACE,
    DOUBLY_NESTED_TUPLE_SPACE,
    NESTED_DICT_SPACE,
    NESTED_TUPLE_SPACE,
    NUM_ENVS,
    TWO_BOX_TUPLE_SPACE,
)
from tests.dev_wrappers.utils import TestingEnv

try:
    from gym.wrappers import observations_dtype_v0
except ImportError:
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize(
    ("env", "args"),
    [
        (gym.make("CartPole-v1", new_step_api=True), np.dtype("int32")),
        (gym.make("CartPole-v1", new_step_api=True), np.dtype("float32")),
    ],
)
def test_observation_dtype_v0(env, args):
    """Test correct dtype is applied to observation."""
    wrapped_env = observations_dtype_v0(env, args)
    wrapped_env.reset()
    obs, _, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    assert obs.dtype == args


@pytest.mark.parametrize(
    ("env", "args"),
    [
        (
            gym.vector.make("CartPole-v1", num_envs=NUM_ENVS, new_step_api=True),
            np.dtype("int32"),
        ),
        (
            gym.vector.make("CartPole-v1", num_envs=NUM_ENVS, new_step_api=True),
            np.dtype("float32"),
        ),
    ],
)
def test_observation_dtype_v0_within_vector(env, args):
    """Test correct dtype is applied to observation in vectorized envs."""
    wrapped_env = observations_dtype_v0(env, args)
    wrapped_env.reset()
    observations, _, _, _, _ = wrapped_env.step(
        [DISCRETE_ACTION for _ in range(NUM_ENVS)]
    )

    for obs in observations:
        assert obs.dtype == args


@pytest.mark.parametrize(
    ("env", "args"),
    [
        (TestingEnv(observation_space=DICT_SPACE), {"box": np.dtype("float32")}),
        (
            TestingEnv(observation_space=DICT_SPACE),
            {"box": np.dtype("float32"), "box2": np.dtype("float32")},
        ),
        (
            TestingEnv(observation_space=DICT_SPACE),
            {"box": np.dtype("uint8"), "box2": np.dtype("uint8")},
        ),
    ],
)
def test_observation_dtype_v0_dict(env, args):
    """Test correct dtype is applied to observation."""
    wrapped_env = observations_dtype_v0(env, args)
    wrapped_env.reset()
    obs, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    for subspace in obs:
        if subspace in args:
            assert obs[subspace].dtype == args[subspace]


@pytest.mark.parametrize(
    ("env", "args"),
    [
        (
            TestingEnv(observation_space=NESTED_DICT_SPACE),
            {"nested": {"nested": np.dtype("int32")}},
        ),
        (
            TestingEnv(observation_space=NESTED_DICT_SPACE),
            {"box": np.dtype("uint8"), "nested": {"nested": np.dtype("int32")}},
        ),
        (
            TestingEnv(observation_space=DOUBLY_NESTED_DICT_SPACE),
            {"nested": {"nested": {"nested": np.dtype("int32")}}},
        ),
    ],
)
def test_observation_dtype_v0_nested_dict(env, args):
    """Test correct dtype is applied to observation."""
    wrapped_env = observations_dtype_v0(env, args)
    wrapped_env.reset()
    obs, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    if "box" in args:
        assert obs["box"].dtype == args["box"]

    dict_subspace = obs["nested"]
    dict_args = args["nested"]
    while isinstance(dict_subspace, OrderedDict):
        dict_subspace = dict_subspace["nested"]
        dict_args = dict_args["nested"]
    assert dict_subspace.dtype == dict_args


@pytest.mark.parametrize(
    ("env", "args"),
    [
        (TestingEnv(observation_space=TWO_BOX_TUPLE_SPACE), [None, np.dtype("int32")]),
        (TestingEnv(observation_space=TWO_BOX_TUPLE_SPACE), [np.dtype("int32"), None]),
        (
            TestingEnv(observation_space=TWO_BOX_TUPLE_SPACE),
            [np.dtype("int32"), np.dtype("int32")],
        ),
    ],
)
def test_observation_dtype_v0_tuple(env, args):
    """Test correct dtype is applied to observation."""
    wrapped_env = observations_dtype_v0(env, args)
    wrapped_env.reset()
    obs, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    for subspace, arg in zip(obs, args):
        if arg:
            assert subspace.dtype == arg


@pytest.mark.parametrize(
    ("env", "args"),
    [
        (
            TestingEnv(observation_space=NESTED_TUPLE_SPACE),
            [None, [None, np.dtype("float32")]],
        ),
        (
            TestingEnv(observation_space=NESTED_TUPLE_SPACE),
            [np.dtype("float32"), [None, np.dtype("float32")]],
        ),
        (
            TestingEnv(observation_space=DOUBLY_NESTED_TUPLE_SPACE),
            [None, [None, [None, np.dtype("float32")]]],
        ),
    ],
)
def test_observation_dtype_v0_nested_tuple(env, args):
    """Test correct dtype is applied to observation."""
    wrapped_env = observations_dtype_v0(env, args)
    wrapped_env.reset()
    obs, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    if args[0]:
        assert obs[0].dtype == args[0]

    tuple_subspace = obs[-1]
    tuple_args = args[-1]
    while isinstance(tuple_subspace[-1], tuple):
        tuple_subspace = tuple_subspace[-1]
        tuple_args = tuple_args[-1]
    assert tuple_subspace[-1].dtype == tuple_args[-1]
