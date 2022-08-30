import pytest

import gym
from tests.dev_wrappers.mock_data import (
    BOX_SPACE,
    DICT_SPACE,
    DOUBLY_NESTED_DICT_SPACE,
    NESTED_DICT_SPACE,
    NEW_BOX_DIM,
    NEW_BOX_DIM_IMPOSSIBLE,
    NUM_ENVS,
    NUM_STEPS,
    SEED,
    TUPLE_SPACE,
)
from tests.dev_wrappers.utils import TestingEnv

try:
    from gym.wrappers import ReshapeObservationsV0
except ImportError:
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize(
    ("env", "args"),
    [
        (TestingEnv(observation_space=BOX_SPACE), NEW_BOX_DIM),
        (
            gym.make("CarRacing-v2", disable_env_checker=True),
            (96, 48, 6),
        ),  # Box(0, 255, (96, 96, 3), uint8)
        (
            gym.vector.make(
                "CarRacing-v2", num_envs=NUM_ENVS, disable_env_checker=True
            ),  # Box(0, 255, (NUM_ENVS, 96, 96, 3), uint8)
            (96, 96, 3 * NUM_ENVS),
        ),
    ],
)
def test_reshape_observations_v0_box(env, args):
    """Test correct reshaping of box observation spaces."""
    wrapped_env = ReshapeObservationsV0(env, args)
    wrapped_env.reset(seed=SEED)

    assert wrapped_env.observation_space.shape == args

    for _ in range(NUM_STEPS):
        action = wrapped_env.action_space.sample()
        obs, *res = wrapped_env.step(action)
        assert obs in wrapped_env.observation_space


def test_reshape_observations_v0_box_impossible():
    """Test wrong new shape raises ValueError.

    A wrong new shape is a shape that can not be
    obtained from the original shape.
    """
    env = TestingEnv(observation_space=BOX_SPACE)

    with pytest.raises(ValueError):
        ReshapeObservationsV0(env, NEW_BOX_DIM_IMPOSSIBLE)


@pytest.mark.parametrize(
    ("env", "args"),
    [
        (TestingEnv(observation_space=DICT_SPACE), {"box": NEW_BOX_DIM}),
        (TestingEnv(observation_space=DICT_SPACE), {}),
    ],
)
def test_reshape_observations_v0_dict(env, args):
    """Test reshaping `Dict` observation spaces.

    Tests whether `Dict` observation spaces are correctly reshaped.
    Expected behaviour is that the reshape observation
    space matches the shape provided in `args`.
    """
    wrapped_env = ReshapeObservationsV0(env, args)
    wrapped_env.reset(seed=SEED)

    for k in wrapped_env.observation_space.keys():
        if k in args:
            assert wrapped_env.observation_space[k].shape == args[k]
        else:
            assert (
                wrapped_env.observation_space[k].shape == env.observation_space[k].shape
            )


@pytest.mark.parametrize(
    ("env", "args"),
    [
        (
            TestingEnv(observation_space=NESTED_DICT_SPACE),
            {"nested": {"nested": NEW_BOX_DIM}},
        ),
        (
            TestingEnv(observation_space=DOUBLY_NESTED_DICT_SPACE),
            {"nested": {"nested": {"nested": NEW_BOX_DIM}}},
        ),
    ],
)
def test_reshape_observations_v0_nested_dict(env, args):
    """Test reshaping nested `Dict` observation spaces.

    Tests whether nested `Dict` observation spaces are correctly reshaped.
    Expected behaviour is that the reshape observation
    space matches the shape provided in `args`.
    """
    wrapped_env = ReshapeObservationsV0(env, args)
    wrapped_env.reset(seed=SEED)

    nested_arg = args["nested"]
    nested_space = wrapped_env.observation_space["nested"]
    while isinstance(nested_arg, dict):
        nested_arg = nested_arg["nested"]
        nested_space = nested_space["nested"]

    assert nested_space.shape == nested_arg


def test_reshape_observations_v0_tuple():
    """Test reshaping `Tuple` observation spaces.

    Tests whether `Tuple` observation spaces are correctly reshaped.
    Expected behaviour is that the reshape observation
    space matches the shape provided.
    """
    env = TestingEnv(observation_space=TUPLE_SPACE)
    args = [None, NEW_BOX_DIM]

    wrapped_env = ReshapeObservationsV0(env, args)
    wrapped_env.reset(seed=SEED)

    for i, arg in enumerate(args):
        if arg:
            assert wrapped_env.observation_space[i].shape == arg
        else:
            assert (
                wrapped_env.observation_space[i].shape == env.observation_space[i].shape
            )
