"""Finds all the specs that we can test with"""
from typing import List, Optional

import numpy as np

import gym
from gym import logger
from gym.envs.registration import EnvSpec


def try_make_env(env_spec: EnvSpec) -> Optional[gym.Env]:
    """Tries to make the environment showing if it is possible.

    Warning the environments have no wrappers, including time limit and order enforcing.
    """
    # To avoid issues with registered environments during testing, we check that the spec entry points are from gym.envs.
    if "gym.envs." in env_spec.entry_point:
        try:
            return env_spec.make(disable_env_checker=True).unwrapped
        except ImportError as e:
            logger.warn(f"Not testing {env_spec.id} due to error: {e}")
    return None


# Tries to make all environment to test with
all_testing_initialised_envs: List[Optional[gym.Env]] = [
    try_make_env(env_spec) for env_spec in gym.envs.registry.values()
]
all_testing_initialised_envs: List[gym.Env] = [
    env for env in all_testing_initialised_envs if env is not None
]

# All testing, mujoco and gym environment specs
all_testing_env_specs: List[EnvSpec] = [
    env.spec for env in all_testing_initialised_envs
]
mujoco_testing_env_specs: List[EnvSpec] = [
    env_spec
    for env_spec in all_testing_env_specs
    if "gym.envs.mujoco" in env_spec.entry_point
]
gym_testing_env_specs: List[EnvSpec] = [
    env_spec
    for env_spec in all_testing_env_specs
    if any(
        f"gym.envs.{ep}" in env_spec.entry_point
        for ep in ["box2d", "classic_control", "toy_text"]
    )
]
# TODO, add minimum testing env spec in testing
minimum_testing_env_specs = [
    env_spec
    for env_spec in [
        "CartPole-v1",
        "MountainCarContinuous-v0",
        "LunarLander-v2",
        "LunarLanderContinuous-v2",
        "CarRacing-v2",
        "Blackjack-v1",
        "Reacher-v4",
    ]
    if env_spec in all_testing_env_specs
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
