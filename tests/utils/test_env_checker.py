"""Tests that the `env_checker` runs as expects and all errors are possible."""
import re

import numpy as np
import pytest

import gym
from gym import spaces
from gym.utils.env_checker import (
    check_env,
    check_reset_info,
    check_reset_options,
    check_reset_seed,
)
from tests.testing_env import GenericTestEnv


@pytest.mark.parametrize(
    "env",
    [
        gym.make("CartPole-v1", disable_env_checker=True),
        gym.make("MountainCar-v0", disable_env_checker=True),
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
    with pytest.warns(None) as warnings:
        check_env(env)

    assert len(warnings) == 0, [warning.message for warning in warnings]


def _no_super_reset(self, seed=None, return_info=False, options=None):
    self.np_random.random()  # generates a new prng
    # generate seed deterministic result
    self.observation_space.seed(0)
    return self.observation_space.sample()


def _super_reset_fixed(self, seed=None, return_info=False, options=None):
    # Call super that ignores the seed passed, use fixed seed
    super(GenericTestEnv, self).reset(seed=1)
    # deterministic output
    self.observation_space._np_random = self.np_random
    return self.observation_space.sample()


def _reset_default_seed(
    self: GenericTestEnv, seed="Error", return_info=False, options=None
):
    super(GenericTestEnv, self).reset(seed=seed)
    self.observation_space._np_random = self.np_random
    return self.observation_space.sample()


@pytest.mark.parametrize(
    "test,func,message",
    [
        [
            gym.error.Error,
            lambda self: self.observation_space.sample(),
            "The `reset` method does not provide the `seed` keyword argument",
        ],
        [
            AssertionError,
            lambda self, seed, *_: self.observation_space.sample(),
            "`env.reset(seed=123)` is not deterministic as the observations are not equivalent",
        ],
        [
            AssertionError,
            _no_super_reset,
            "Mostly likely the environment reset function does not call `super().reset(seed=seed)` as the random generates are not same when the same seeds are passed to `env.reset`.",
        ],
        [
            AssertionError,
            _super_reset_fixed,
            "Mostly likely the environment reset function does not call `super().reset(seed=seed)` as the random generates are not different when different seeds are passed to `env.reset`.",
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


def _reset_return_info_type(self, seed=None, return_info=False, options=None):
    if return_info:
        return [1, 2]
    else:
        return self.observation_space.sample()


def _reset_return_info_length(self, seed=None, return_info=False, options=None):
    if return_info:
        return 1, 2, 3
    else:
        return self.observation_space.sample()


def _return_info_obs_outside(self, seed=None, return_info=False, options=None):
    if return_info:
        return self.observation_space.sample() + self.observation_space.high, {}
    else:
        return self.observation_space.sample()


def _return_info_not_dict(self, seed=None, return_info=False, options=None):
    if return_info:
        return self.observation_space.sample(), ["key", "value"]
    else:
        return self.observation_space.sample()


@pytest.mark.parametrize(
    "test,func,message",
    [
        [
            gym.error.Error,
            lambda self, *_: self.observation_space.sample(),
            "The `reset` method does not provide the `return_info` keyword argument",
        ],
        [
            AssertionError,
            _reset_return_info_type,
            "Calling the reset method with `return_info=True` did not return a tuple, actual type: <class 'list'>",
        ],
        [
            AssertionError,
            _reset_return_info_length,
            "Calling the reset method with `return_info=True` did not return a 2-tuple, actual length: 3",
        ],
        [
            AssertionError,
            _return_info_obs_outside,
            "The first element returned by `env.reset(return_info=True)` is not within the observation space",
        ],
        [
            AssertionError,
            _return_info_not_dict,
            "The second element returned by `env.reset(return_info=True)` was not a dictionary",
        ],
    ],
)
def test_check_reset_info(test, func: callable, message: str):
    """Tests the check reset info function works as expected."""
    if test is UserWarning:
        with pytest.warns(
            UserWarning, match=f"^\\x1b\\[33mWARN: {re.escape(message)}\\x1b\\[0m$"
        ):
            check_reset_info(GenericTestEnv(reset_fn=func))
    else:
        with pytest.raises(test, match=f"^{re.escape(message)}$"):
            check_reset_info(GenericTestEnv(reset_fn=func))


def test_check_reset_options():
    """Tests the check_reset_options function."""

    with pytest.raises(
        gym.error.Error,
        match=re.escape(
            "The `reset` method does not provide the `options` keyword argument"
        ),
    ):
        check_reset_options(GenericTestEnv(reset_fn=lambda self: 0))


@pytest.mark.parametrize(
    "env,message",
    [
        [
            "Error",
            "Your environment must inherit from the gym.Env class https://www.gymlibrary.ml/content/environment_creation/",
        ],
        [
            GenericTestEnv(action_space=None),
            "You must specify a action space. https://www.gymlibrary.ml/content/environment_creation/",
        ],
        [
            GenericTestEnv(observation_space=None),
            "You must specify an observation space. https://www.gymlibrary.ml/content/environment_creation/",
        ],
    ],
)
def test_check_env(env: gym.Env, message: str):
    """Tests the check_env function works as expected."""
    with pytest.raises(AssertionError, match=f"^{re.escape(message)}$"):
        check_env(env)
