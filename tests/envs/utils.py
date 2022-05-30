"""Finds all the specs that we can test with"""

from typing import Optional, Union, Dict, Tuple

import numpy as np

import gym
from gym.envs.registration import EnvSpec


def if_test_env_spec(env_spec: EnvSpec) -> Optional[gym.Env]:
    """Tries to make the environment showing if it is possible."""
    try:
        return env_spec.make(disable_env_checker=True)
    except ImportError:
        return None


testing_envs = filter(lambda x: x is not None, [
    if_test_env_spec(env_spec)
    for env_spec in gym.envs.registry.values()
])
mujoco_testing_envs = [
    env
    for env in testing_envs
    if "mujoco" in env.spec.entry_point
]
gym_testing_envs = [
    env
    for env in testing_envs
    for gym_entry_point in ["box2d", "classic_control", "toy_text"]
    if gym_entry_point in env.spec.entry_point
]


def assert_equals(a, b, prefix=None):
    """Assert equality of data structures `a` and `b`.

    Args:
        a: first data structure
        b: second data structure
        prefix: prefix for failed assertion message for types and dicts
    """
    assert type(a) == type(b), f"{prefix}Differing types: {a} and {b}"
    if isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), f"{prefix}Key sets differ: {a} and {b}"

        for k in a.keys():
            v_a = a[k]
            v_b = b[k]
            assert_equals(v_a, v_b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, tuple):
        for elem_from_a, elem_from_b in zip(a, b):
            assert_equals(elem_from_a, elem_from_b)
    else:
        assert a == b
