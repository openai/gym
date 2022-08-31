"""Tests that the `env_checker` runs as expects and all errors are possible."""
import re
import warnings
from typing import Tuple, Union

import numpy as np
import pytest

import gym
from gym import spaces
from gym.core import ObsType
from gym.utils.env_checker import (
    check_env,
    check_reset_options,
    check_reset_return_info_deprecation,
    check_reset_return_type,
    check_reset_seed,
    check_seed_deprecation,
)
from tests.testing_env import GenericTestEnv


@pytest.mark.parametrize(
    "env",
    [
        gym.make("CartPole-v1", disable_env_checker=True).unwrapped,
        gym.make("MountainCar-v0", disable_env_checker=True).unwrapped,
        GenericTestEnv(
            observation_space=spaces.Dict(
                a=spaces.Discrete(10), b=spaces.Box(np.zeros(2), np.ones(2))
            )
        ),
        GenericTestEnv(
            observation_space=spaces.Tuple(
                [spaces.Discrete(10), spaces.Box(np.zeros(2), np.ones(2))]
            )
        ),
        GenericTestEnv(
            observation_space=spaces.Dict(
                a=spaces.Tuple(
                    [spaces.Discrete(10), spaces.Box(np.zeros(2), np.ones(2))]
                ),
                b=spaces.Box(np.zeros(2), np.ones(2)),
            )
        ),
    ],
)
def test_no_error_warnings(env):
    """A full version of this test with all gym envs is run in tests/envs/test_envs.py."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env)

    assert len(caught_warnings) == 0, [warning.message for warning in caught_warnings]


def _no_super_reset(self, seed=None, options=None):
    self.np_random.random()  # generates a new prng
    # generate seed deterministic result
    self.observation_space.seed(0)
    return self.observation_space.sample(), {}


def _super_reset_fixed(self, seed=None, options=None):
    # Call super that ignores the seed passed, use fixed seed
    super(GenericTestEnv, self).reset(seed=1)
    # deterministic output
    self.observation_space._np_random = self.np_random
    return self.observation_space.sample(), {}


def _reset_default_seed(self: GenericTestEnv, seed="Error", options=None):
    super(GenericTestEnv, self).reset(seed=seed)
    self.observation_space._np_random = (  # pyright: ignore [reportPrivateUsage]
        self.np_random
    )
    return self.observation_space.sample(), {}


@pytest.mark.parametrize(
    "test,func,message",
    [
        [
            gym.error.Error,
            lambda self: (self.observation_space.sample(), {}),
            "The `reset` method does not provide a `seed` or `**kwargs` keyword argument.",
        ],
        [
            AssertionError,
            lambda self, seed, *_: (self.observation_space.sample(), {}),
            "Expects the random number generator to have been generated given a seed was passed to reset. Mostly likely the environment reset function does not call `super().reset(seed=seed)`.",
        ],
        [
            AssertionError,
            _no_super_reset,
            "Mostly likely the environment reset function does not call `super().reset(seed=seed)` as the random generates are not same when the same seeds are passed to `env.reset`.",
        ],
        [
            AssertionError,
            _super_reset_fixed,
            "Mostly likely the environment reset function does not call `super().reset(seed=seed)` as the random number generators are not different when different seeds are passed to `env.reset`.",
        ],
        [
            UserWarning,
            _reset_default_seed,
            "The default seed argument in reset should be `None`, otherwise the environment will by default always be deterministic. Actual default: Error",
        ],
    ],
)
def test_check_reset_seed(test, func: callable, message: str):
    """Tests the check reset seed function works as expected."""
    if test is UserWarning:
        with pytest.warns(
            UserWarning, match=f"^\\x1b\\[33mWARN: {re.escape(message)}\\x1b\\[0m$"
        ):
            check_reset_seed(GenericTestEnv(reset_fn=func))
    else:
        with pytest.raises(test, match=f"^{re.escape(message)}$"):
            check_reset_seed(GenericTestEnv(reset_fn=func))


def _deprecated_return_info(
    self, return_info: bool = False
) -> Union[Tuple[ObsType, dict], ObsType]:
    """function to simulate the signature and behavior of a `reset` function with the deprecated `return_info` optional argument"""
    if return_info:
        return self.observation_space.sample(), {}
    else:
        return self.observation_space.sample()


def _reset_var_keyword_kwargs(self, kwargs):
    return self.observation_space.sample(), {}


def _reset_return_info_type(self, seed=None, options=None):
    """Returns a `list` instead of a `tuple`. This function is used to make sure `env_checker` correctly
    checks that the return type of `env.reset()` is a `tuple`"""
    return [self.observation_space.sample(), {}]


def _reset_return_info_length(self, seed=None, options=None):
    return 1, 2, 3


def _return_info_obs_outside(self, seed=None, options=None):
    return self.observation_space.sample() + self.observation_space.high, {}


def _return_info_not_dict(self, seed=None, options=None):
    return self.observation_space.sample(), ["key", "value"]


@pytest.mark.parametrize(
    "test,func,message",
    [
        [
            AssertionError,
            _reset_return_info_type,
            "The result returned by `env.reset()` was not a tuple of the form `(obs, info)`, where `obs` is a observation and `info` is a dictionary containing additional information. Actual type: `<class 'list'>`",
        ],
        [
            AssertionError,
            _reset_return_info_length,
            "Calling the reset method did not return a 2-tuple, actual length: 3",
        ],
        [
            AssertionError,
            _return_info_obs_outside,
            "The first element returned by `env.reset()` is not within the observation space.",
        ],
        [
            AssertionError,
            _return_info_not_dict,
            "The second element returned by `env.reset()` was not a dictionary, actual type: <class 'list'>",
        ],
    ],
)
def test_check_reset_return_type(test, func: callable, message: str):
    """Tests the check `env.reset()` function has a correct return type."""

    with pytest.raises(test, match=f"^{re.escape(message)}$"):
        check_reset_return_type(GenericTestEnv(reset_fn=func))


@pytest.mark.parametrize(
    "test,func,message",
    [
        [
            UserWarning,
            _deprecated_return_info,
            "`return_info` is deprecated as an optional argument to `reset`. `reset`"
            "should now always return `obs, info` where `obs` is an observation, and `info` is a dictionary"
            "containing additional information.",
        ],
    ],
)
def test_check_reset_return_info_deprecation(test, func: callable, message: str):
    """Tests that return_info has been correct deprecated as an argument to `env.reset()`."""

    with pytest.warns(test, match=f"^\\x1b\\[33mWARN: {re.escape(message)}\\x1b\\[0m$"):
        check_reset_return_info_deprecation(GenericTestEnv(reset_fn=func))


def test_check_seed_deprecation():
    """Tests that `check_seed_deprecation()` throws a warning if `env.seed()` has not been removed."""

    message = """Official support for the `seed` function is dropped. Standard practice is to reset gym environments using `env.reset(seed=<desired seed>)`"""

    env = GenericTestEnv()

    def seed(seed):
        return

    with pytest.warns(
        UserWarning, match=f"^\\x1b\\[33mWARN: {re.escape(message)}\\x1b\\[0m$"
    ):
        env.seed = seed
        assert callable(env.seed)
        check_seed_deprecation(env)

    with warnings.catch_warnings(record=True) as caught_warnings:
        env.seed = []
        check_seed_deprecation(env)
        env.seed = 123
        check_seed_deprecation(env)
        del env.seed
        check_seed_deprecation(env)
        assert len(caught_warnings) == 0


def test_check_reset_options():
    """Tests the check_reset_options function."""
    with pytest.raises(
        gym.error.Error,
        match=re.escape(
            "The `reset` method does not provide an `options` or `**kwargs` keyword argument"
        ),
    ):
        check_reset_options(GenericTestEnv(reset_fn=lambda self: (0, {})))


@pytest.mark.parametrize(
    "env,message",
    [
        [
            "Error",
            "The environment must inherit from the gym.Env class. See https://www.gymlibrary.dev/content/environment_creation/ for more info.",
        ],
        [
            GenericTestEnv(action_space=None),
            "The environment must specify an action space. See https://www.gymlibrary.dev/content/environment_creation/ for more info.",
        ],
        [
            GenericTestEnv(observation_space=None),
            "The environment must specify an observation space. See https://www.gymlibrary.dev/content/environment_creation/ for more info.",
        ],
    ],
)
def test_check_env(env: gym.Env, message: str):
    """Tests the check_env function works as expected."""
    with pytest.raises(AssertionError, match=f"^{re.escape(message)}$"):
        check_env(env)
