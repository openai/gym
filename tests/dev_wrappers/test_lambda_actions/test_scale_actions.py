"""Test suite for ScaleActionsV0."""
from collections import OrderedDict
from typing import Sequence

import numpy as np
import pytest

import gym
from tests.dev_wrappers.mock_data import (
    BOX_HIGH,
    BOX_LOW,
    DISCRETE_ACTION,
    DOUBLY_NESTED_DICT_SPACE,
    DOUBLY_NESTED_TUPLE_SPACE,
    NESTED_DICT_SPACE,
    NESTED_TUPLE_SPACE,
    NEW_BOX_HIGH,
    NEW_BOX_LOW,
    NUM_ENVS,
    SEED,
    TUPLE_SPACE,
)
from tests.dev_wrappers.utils import TestingEnv

try:
    from gym.wrappers import ScaleActionsV0
except ImportError:
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize(
    ("env", "args", "action", "scaled_action"),
    [
        (
            # BipedalWalker action space: Box(-1.0, 1.0, (4,), float32)
            gym.make("BipedalWalker-v3", new_step_api=True),
            (-0.5, 0.5),
            np.array([1, 1, 1, 1]),
            np.array([0.5, 0.5, 0.5, 0.5]),
        ),
    ],
)
def test_scale_actions_v0_box(env, args, action, scaled_action):
    """Test action rescaling.

    Scale action wrapper allow to rescale action
    to a new range.
    Supposed the old action space is
    `Box(-1, 1, (1,))` and we rescale to
    `Box(-0.5, 0.5, (1,))`, an action  with value
    `0.5` will have the same effect of an action with value
    `1.0` on the unwrapped env.
    """
    env.reset(seed=SEED)
    obs, _, _, _, _ = env.step(action)

    env.reset(seed=SEED)
    wrapped_env = ScaleActionsV0(env, args)

    obs_scaled, _, _, _, _ = wrapped_env.step(scaled_action)

    assert np.alltrue(obs == obs_scaled)


@pytest.mark.parametrize(
    ("env", "args", "action", "scaled_action"),
    [
        (
            gym.vector.make("BipedalWalker-v3", num_envs=NUM_ENVS, new_step_api=True),
            (-0.5, 0.5),
            np.tile(np.array([1, 1, 1, 1]), (NUM_ENVS, 1)),
            np.tile(np.array([0.5, 0.5, 0.5, 0.5]), (NUM_ENVS, 1)),
        ),
    ],
)
def test_scale_action_v0_within_vector(env, args, action, scaled_action):
    """Tests scale action in vectorized environments.

    Tests if action is correctly rescaled in vectorized environment.
    """
    env.reset(seed=SEED)
    obs, _, _, _, _ = env.step(action)

    env.reset(seed=SEED)
    wrapped_env = ScaleActionsV0(env, args)
    obs_scaled, _, _, _, _ = wrapped_env.step(scaled_action)

    assert np.alltrue(obs == obs_scaled)


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            TestingEnv(action_space=NESTED_DICT_SPACE),
            {
                "box": (NEW_BOX_LOW, NEW_BOX_HIGH),
                "nested": {"nested": (BOX_LOW, BOX_HIGH)},
            },
            {
                "box": NEW_BOX_HIGH,
                "discrete": DISCRETE_ACTION,
                "nested": {"nested": BOX_HIGH},
            },
        ),
        (
            TestingEnv(action_space=DOUBLY_NESTED_DICT_SPACE),
            {
                "box": (NEW_BOX_LOW, NEW_BOX_HIGH),
                "nested": {"nested": {"nested": (BOX_LOW, BOX_HIGH)}},
            },
            {
                "box": NEW_BOX_HIGH,
                "discrete": DISCRETE_ACTION,
                "nested": {"nested": {"nested": BOX_HIGH}},
            },
        ),
    ],
)
def test_scale_actions_v0_nested_dict(env, args, action):
    """Test action rescaling for nested `Dict` action spaces."""
    wrapped_env = ScaleActionsV0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert executed_actions["box"] == BOX_HIGH

    nested_action = executed_actions["nested"]
    while isinstance(nested_action, OrderedDict):
        nested_action = nested_action["nested"]
    assert nested_action == BOX_HIGH


def test_scale_actions_v0_tuple():
    """Test action rescaling for `Tuple` action spaces."""
    env = TestingEnv(action_space=TUPLE_SPACE)
    args = [None, (NEW_BOX_LOW, NEW_BOX_HIGH)]
    action = [DISCRETE_ACTION, NEW_BOX_HIGH]

    wrapped_env = ScaleActionsV0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert executed_actions[0] == action[0]
    assert executed_actions[1] == BOX_HIGH


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            TestingEnv(action_space=NESTED_TUPLE_SPACE),
            [None, [None, (NEW_BOX_LOW, NEW_BOX_HIGH)]],
            [BOX_HIGH, [DISCRETE_ACTION, NEW_BOX_HIGH]],
        ),
        (
            TestingEnv(action_space=DOUBLY_NESTED_TUPLE_SPACE),
            [None, [None, [None, (NEW_BOX_LOW, NEW_BOX_HIGH)]]],
            [BOX_HIGH, [DISCRETE_ACTION, [DISCRETE_ACTION, NEW_BOX_HIGH]]],
        ),
    ],
)
def test_scale_actions_v0_nested_tuple(env, args, action):
    """Test action rescaling for nested `Tuple` action spaces."""
    wrapped_env = ScaleActionsV0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    nested_action = executed_actions[-1]
    while isinstance(nested_action, Sequence):
        nested_action = nested_action[-1]

    assert executed_actions[0] == BOX_HIGH
    assert nested_action == BOX_HIGH
