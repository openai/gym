import re
import warnings
from typing import Dict, Union

import numpy as np
import pytest

import gym
from gym import spaces
from gym.utils.passive_env_checker import (
    check_action_space,
    check_obs,
    check_observation_space,
    env_render_passive_checker,
    env_reset_passive_checker,
    env_step_passive_checker,
)
from tests.testing_env import GenericTestEnv


def _modify_space(space: spaces.Space, attribute: str, value):
    setattr(space, attribute, value)
    return space


@pytest.mark.parametrize(
    "test,space,message",
    [
        [
            AssertionError,
            "error",
            "observation space does not inherit from `gym.spaces.Space`, actual type: <class 'str'>",
        ],
        # ===== Check box observation space ====
        [
            UserWarning,
            spaces.Box(np.zeros((5, 5, 1)), 255 * np.ones((5, 5, 1)), dtype=np.int32),
            "It seems a Box observation space is an image but the `dtype` is not `np.uint8`, actual type: int32. If the Box observation space is not an image, we recommend flattening the observation to have only a 1D vector.",
        ],
        [
            UserWarning,
            spaces.Box(np.ones((2, 2, 1)), 255 * np.ones((2, 2, 1)), dtype=np.uint8),
            "It seems a Box observation space is an image but the upper and lower bounds are not in [0, 255]. Generally, CNN policies assume observations are within that range, so you may encounter an issue if the observation values are not.",
        ],
        [
            UserWarning,
            spaces.Box(np.zeros((5, 5, 1)), np.ones((5, 5, 1)), dtype=np.uint8),
            "It seems a Box observation space is an image but the upper and lower bounds are not in [0, 255]. Generally, CNN policies assume observations are within that range, so you may encounter an issue if the observation values are not.",
        ],
        [
            UserWarning,
            spaces.Box(np.zeros((5, 5)), np.ones((5, 5))),
            "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: (5, 5)",
        ],
        [
            UserWarning,
            spaces.Box(np.zeros(5), np.zeros(5)),
            "A Box observation space maximum and minimum values are equal.",
        ],
        [
            UserWarning,
            spaces.Box(np.ones(5), np.zeros(5)),
            "A Box observation space low value is greater than a high value.",
        ],
        [
            AssertionError,
            _modify_space(spaces.Box(np.zeros(2), np.ones(2)), "low", np.zeros(3)),
            "The Box observation space shape and low shape have different shapes, low shape: (3,), box shape: (2,)",
        ],
        [
            AssertionError,
            _modify_space(spaces.Box(np.zeros(2), np.ones(2)), "high", np.ones(3)),
            "The Box observation space shape and high shape have have different shapes, high shape: (3,), box shape: (2,)",
        ],
        # ==== Other observation spaces (Discrete, MultiDiscrete, MultiBinary, Tuple, Dict)
        [
            AssertionError,
            _modify_space(spaces.Discrete(5), "n", -1),
            "Discrete observation space's number of elements must be positive, actual number of elements: -1",
        ],
        [
            AssertionError,
            _modify_space(spaces.MultiDiscrete([2, 2]), "nvec", np.array([2, -1])),
            "Multi-discrete observation space's all nvec elements must be greater than 0, actual nvec: [ 2 -1]",
        ],
        [
            AssertionError,
            _modify_space(spaces.MultiDiscrete([2, 2]), "_shape", (2, 1, 2)),
            "Multi-discrete observation space's shape must be equal to the nvec shape, space shape: (2, 1, 2), nvec shape: (2,)",
        ],
        [
            AssertionError,
            _modify_space(spaces.MultiBinary((2, 2)), "_shape", (2, -1)),
            "Multi-binary observation space's all shape elements must be greater than 0, actual shape: (2, -1)",
        ],
        [
            AssertionError,
            spaces.Tuple([]),
            "An empty Tuple observation space is not allowed.",
        ],
        [
            AssertionError,
            spaces.Dict(),
            "An empty Dict observation space is not allowed.",
        ],
    ],
)
def test_check_observation_space(test, space, message: str):
    """Tests the check observation space."""
    if test is UserWarning:
        with pytest.warns(
            UserWarning, match=f"^\\x1b\\[33mWARN: {re.escape(message)}\\x1b\\[0m$"
        ):
            check_observation_space(space)
    else:
        with warnings.catch_warnings(record=True) as caught_warnings:
            with pytest.raises(test, match=f"^{re.escape(message)}$"):
                check_observation_space(space)
        assert len(caught_warnings) == 0


@pytest.mark.parametrize(
    "test,space,message",
    [
        [
            AssertionError,
            "error",
            "action space does not inherit from `gym.spaces.Space`, actual type: <class 'str'>",
        ],
        # ===== Check box observation space ====
        [
            UserWarning,
            spaces.Box(np.zeros(5), np.zeros(5)),
            "A Box action space maximum and minimum values are equal.",
        ],
        [
            UserWarning,
            spaces.Box(np.ones(5), np.zeros(5)),
            "A Box action space low value is greater than a high value.",
        ],
        [
            AssertionError,
            _modify_space(spaces.Box(np.zeros(2), np.ones(2)), "low", np.zeros(3)),
            "The Box action space shape and low shape have have different shapes, low shape: (3,), box shape: (2,)",
        ],
        [
            AssertionError,
            _modify_space(spaces.Box(np.zeros(2), np.ones(2)), "high", np.ones(3)),
            "The Box action space shape and high shape have different shapes, high shape: (3,), box shape: (2,)",
        ],
        # ==== Other observation spaces (Discrete, MultiDiscrete, MultiBinary, Tuple, Dict)
        [
            AssertionError,
            _modify_space(spaces.Discrete(5), "n", -1),
            "Discrete action space's number of elements must be positive, actual number of elements: -1",
        ],
        [
            AssertionError,
            _modify_space(spaces.MultiDiscrete([2, 2]), "_shape", (2, -1)),
            "Multi-discrete action space's shape must be equal to the nvec shape, space shape: (2, -1), nvec shape: (2,)",
        ],
        [
            AssertionError,
            _modify_space(spaces.MultiBinary((2, 2)), "_shape", (2, -1)),
            "Multi-binary action space's all shape elements must be greater than 0, actual shape: (2, -1)",
        ],
        [
            AssertionError,
            spaces.Tuple([]),
            "An empty Tuple action space is not allowed.",
        ],
        [AssertionError, spaces.Dict(), "An empty Dict action space is not allowed."],
    ],
)
def test_check_action_space(
    test: Union[UserWarning, type], space: spaces.Space, message: str
):
    """Tests the check action space function."""
    if test is UserWarning:
        with pytest.warns(
            UserWarning, match=f"^\\x1b\\[33mWARN: {re.escape(message)}\\x1b\\[0m$"
        ):
            check_action_space(space)
    else:
        with warnings.catch_warnings(record=True) as caught_warnings:
            with pytest.raises(test, match=f"^{re.escape(message)}$"):
                check_action_space(space)
        assert len(caught_warnings) == 0


@pytest.mark.parametrize(
    "test,obs,obs_space,message",
    [
        [
            UserWarning,
            3,
            spaces.Discrete(2),
            "The obs returned by the `testing()` method is not within the observation space.",
        ],
        [
            UserWarning,
            np.uint8(0),
            spaces.Discrete(1),
            "The obs returned by the `testing()` method should be an int or np.int64, actual type: <class 'numpy.uint8'>",
        ],
        [
            UserWarning,
            [0, 1],
            spaces.Tuple([spaces.Discrete(1), spaces.Discrete(2)]),
            "The obs returned by the `testing()` method was expecting a tuple, actual type: <class 'list'>",
        ],
        [
            AssertionError,
            (1, 2, 3),
            spaces.Tuple([spaces.Discrete(1), spaces.Discrete(2)]),
            "The obs returned by the `testing()` method length is not same as the observation space length, obs length: 3, space length: 2",
        ],
        [
            AssertionError,
            {1, 2, 3},
            spaces.Dict(a=spaces.Discrete(1), b=spaces.Discrete(2)),
            "The obs returned by the `testing()` method must be a dict, actual type: <class 'set'>",
        ],
        [
            AssertionError,
            {"a": 1, "c": 2},
            spaces.Dict(a=spaces.Discrete(1), b=spaces.Discrete(2)),
            "The obs returned by the `testing()` method observation keys is not same as the observation space keys, obs keys: ['a', 'c'], space keys: ['a', 'b']",
        ],
    ],
)
def test_check_obs(test, obs, obs_space: spaces.Space, message: str):
    """Tests the check observations function."""
    if test is UserWarning:
        with pytest.warns(
            UserWarning, match=f"^\\x1b\\[33mWARN: {re.escape(message)}\\x1b\\[0m$"
        ):
            check_obs(obs, obs_space, "testing")
    else:
        with warnings.catch_warnings(record=True) as caught_warnings:
            with pytest.raises(test, match=f"^{re.escape(message)}$"):
                check_obs(obs, obs_space, "testing")
        assert len(caught_warnings) == 0


def _reset_no_seed(self, options=None):
    return self.observation_space.sample(), {}


def _reset_seed_default(self, seed="error", options=None):
    return self.observation_space.sample(), {}


def _reset_no_option(self, seed=None):
    return self.observation_space.sample(), {}


def _make_reset_results(results):
    def _reset_result(self, seed=None, options=None):
        return results

    return _reset_result


@pytest.mark.parametrize(
    "test,func,message,kwargs",
    [
        [
            UserWarning,
            _reset_no_seed,
            "Future gym versions will require that `Env.reset` can be passed a `seed` instead of using `Env.seed` for resetting the environment random number generator.",
            {},
        ],
        [
            UserWarning,
            _reset_seed_default,
            "The default seed argument in `Env.reset` should be `None`, otherwise the environment will by default always be deterministic. Actual default: seed='error'",
            {},
        ],
        [
            UserWarning,
            _reset_no_option,
            "Future gym versions will require that `Env.reset` can be passed `options` to allow the environment initialisation to be passed additional information.",
            {},
        ],
        [
            UserWarning,
            _make_reset_results([0, {}]),
            "The result returned by `env.reset()` was not a tuple of the form `(obs, info)`, where `obs` is a observation and `info` is a dictionary containing additional information. Actual type: `<class 'list'>`",
            {},
        ],
        [
            AssertionError,
            _make_reset_results((np.array([0], dtype=np.float32), {1, 2})),
            "The second element returned by `env.reset()` was not a dictionary, actual type: <class 'set'>",
            {},
        ],
    ],
)
def test_passive_env_reset_checker(test, func: callable, message: str, kwargs: Dict):
    """Tests the passive env reset check"""
    if test is UserWarning:
        with pytest.warns(
            UserWarning, match=f"^\\x1b\\[33mWARN: {re.escape(message)}\\x1b\\[0m$"
        ):
            env_reset_passive_checker(GenericTestEnv(reset_fn=func), **kwargs)
    else:
        with warnings.catch_warnings(record=True) as caught_warnings:
            with pytest.raises(test, match=f"^{re.escape(message)}$"):
                env_reset_passive_checker(GenericTestEnv(reset_fn=func), **kwargs)
        assert len(caught_warnings) == 0


def _modified_step(
    self, obs=None, reward=0, terminated=False, truncated=False, info=None
):
    if obs is None:
        obs = self.observation_space.sample()
    if info is None:
        info = {}

    if truncated is None:
        return obs, reward, terminated, info
    else:
        return obs, reward, terminated, truncated, info


@pytest.mark.parametrize(
    "test,func,message",
    [
        [
            AssertionError,
            lambda self, _: "error",
            "Expects step result to be a tuple, actual type: <class 'str'>",
        ],
        [
            UserWarning,
            lambda self, _: _modified_step(self, terminated="error", truncated=None),
            "Expects `done` signal to be a boolean, actual type: <class 'str'>",
        ],
        [
            UserWarning,
            lambda self, _: _modified_step(self, terminated="error", truncated=False),
            "Expects `terminated` signal to be a boolean, actual type: <class 'str'>",
        ],
        [
            UserWarning,
            lambda self, _: _modified_step(self, truncated="error"),
            "Expects `truncated` signal to be a boolean, actual type: <class 'str'>",
        ],
        [
            gym.error.Error,
            lambda self, _: (1, 2, 3),
            "Expected `Env.step` to return a four or five element tuple, actual number of elements returned: 3.",
        ],
        [
            UserWarning,
            lambda self, _: _modified_step(self, reward="error"),
            "The reward returned by `step()` must be a float, int, np.integer or np.floating, actual type: <class 'str'>",
        ],
        [
            UserWarning,
            lambda self, _: _modified_step(self, reward=np.nan),
            "The reward is a NaN value.",
        ],
        [
            UserWarning,
            lambda self, _: _modified_step(self, reward=np.inf),
            "The reward is an inf value.",
        ],
        [
            AssertionError,
            lambda self, _: _modified_step(self, info="error"),
            "The `info` returned by `step()` must be a python dictionary, actual type: <class 'str'>",
        ],
    ],
)
def test_passive_env_step_checker(
    test: Union[UserWarning, type], func: callable, message: str
):
    """Tests the passive env step checker."""
    if test is UserWarning:
        with pytest.warns(
            UserWarning, match=f"^\\x1b\\[33mWARN: {re.escape(message)}\\x1b\\[0m$"
        ):
            env_step_passive_checker(GenericTestEnv(step_fn=func), 0)
    else:
        with warnings.catch_warnings(record=True) as caught_warnings:
            with pytest.raises(test, match=f"^{re.escape(message)}$"):
                env_step_passive_checker(GenericTestEnv(step_fn=func), 0)
        assert len(caught_warnings) == 0, caught_warnings


@pytest.mark.parametrize(
    "test,env,message",
    [
        [
            UserWarning,
            GenericTestEnv(metadata={"render_modes": None}),
            "No render modes was declared in the environment (env.metadata['render_modes'] is None or not defined), you may have trouble when calling `.render()`.",
        ],
        [
            UserWarning,
            GenericTestEnv(metadata={"render_modes": "Testing mode"}),
            "Expects the render_modes to be a sequence (i.e. list, tuple), actual type: <class 'str'>",
        ],
        [
            UserWarning,
            GenericTestEnv(
                metadata={"render_modes": ["Testing mode", 1], "render_fps": 1},
            ),
            "Expects all render modes to be strings, actual types: [<class 'str'>, <class 'int'>]",
        ],
        [
            UserWarning,
            GenericTestEnv(
                metadata={"render_modes": ["Testing mode"], "render_fps": None},
                render_mode="Testing mode",
                render_fn=lambda self: 0,
            ),
            "No render fps was declared in the environment (env.metadata['render_fps'] is None or not defined), rendering may occur at inconsistent fps.",
        ],
        [
            UserWarning,
            GenericTestEnv(
                metadata={"render_modes": ["Testing mode"], "render_fps": "fps"}
            ),
            "Expects the `env.metadata['render_fps']` to be an integer or a float, actual type: <class 'str'>",
        ],
        [
            AssertionError,
            GenericTestEnv(
                metadata={"render_modes": [], "render_fps": 30}, render_mode="Test"
            ),
            "With no render_modes, expects the Env.render_mode to be None, actual value: Test",
        ],
        [
            AssertionError,
            GenericTestEnv(
                metadata={"render_modes": ["Testing mode"], "render_fps": 30},
                render_mode="Non mode",
            ),
            "The environment was initialized successfully however with an unsupported render mode. Render mode: Non mode, modes: ['Testing mode']",
        ],
    ],
)
def test_passive_render_checker(test, env: GenericTestEnv, message: str):
    """Tests the passive render checker."""
    if test is UserWarning:
        with pytest.warns(
            UserWarning, match=f"^\\x1b\\[33mWARN: {re.escape(message)}\\x1b\\[0m$"
        ):
            env_render_passive_checker(env)
    else:
        with warnings.catch_warnings(record=True) as caught_warnings:
            with pytest.raises(test, match=f"^{re.escape(message)}$"):
                env_render_passive_checker(env)
        assert len(caught_warnings) == 0
