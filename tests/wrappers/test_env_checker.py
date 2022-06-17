import re

import pytest

import gym
from gym.wrappers.env_checker import PassiveEnvChecker
from tests.envs.utils import all_testing_initialised_envs
from tests.testing_env import GenericTestEnv
from tests.wrappers.utils import has_wrapper


@pytest.mark.parametrize(
    "env, message",
    [
        (
            GenericTestEnv(action_space=None),
            "You must specify a action space. https://www.gymlibrary.ml/content/environment_creation/",
        ),
        (
            GenericTestEnv(action_space="error"),
            "Action space does not inherit from `gym.spaces.Space`, actual type: <class 'str'>",
        ),
        (
            GenericTestEnv(observation_space=None),
            "You must specify an observation space. https://www.gymlibrary.ml/content/environment_creation/",
        ),
        (
            GenericTestEnv(observation_space="error"),
            "Observation space does not inherit from `gym.spaces.Space`, actual type: <class 'str'>",
        ),
    ],
)
def test_initialise_failures(env, message):
    with pytest.raises(AssertionError, match=f"^{re.escape(message)}$"):
        PassiveEnvChecker(env)

    env.close()


def _reset_failure(self, seed=None, return_info=False, options=None):
    return "error"


def _step_failure(self, action):
    return "error"


def test_api_failures():
    env = GenericTestEnv(
        reset_fn=_reset_failure, step_fn=_step_failure, render_modes="error"
    )
    env = PassiveEnvChecker(env)
    assert env.checked_reset is False
    assert env.checked_step is False
    assert env.checked_render is False

    with pytest.raises(
        AssertionError,
        match=r"The obs returned by the `reset\(\)` method must be a numpy array, actually type: <class 'str'>",
    ):
        env.reset()
    assert env.checked_reset

    with pytest.raises(
        AssertionError,
        match="Expects step result to be a tuple, actual type: <class 'str'>",
    ):
        env.step(env.action_space.sample())
    assert env.checked_step

    with pytest.raises(
        AssertionError,
        match=r"Expects the render_modes to be a sequence \(i\.e\. list, tuple\), actual type: <class 'str'>",
    ):
        env.render()
    assert env.checked_render

    env.close()


IGNORE_WARNINGS = [
    r"The environment \w+-v\d is out of date\. You should consider upgrading to version `v\d`\.",
    re.escape(
        "This version of the mujoco environments depends on the mujoco-py bindings, which are no longer maintained and may stop working. Please upgrade to the v4 versions of the environments (which depend on the mujoco python bindings instead), unless you are trying to precisely replicate previous works)."
    ),
]
IGNORE_WARNINGS = [
    f"^\\x1b\\[33mWARN: {warning}\\x1b\\[0m$" for warning in IGNORE_WARNINGS
]


RGB_SPECS = [
    env.spec
    for env in all_testing_initialised_envs
    if "rgb_array" in env.metadata["render_modes"]
]


@pytest.mark.parametrize("spec", RGB_SPECS, ids=[spec.id for spec in RGB_SPECS])
def test_wrapper_passes(spec):
    with pytest.warns(None) as warnings:
        env = gym.make(spec.id, render_mode="rgb_array")
        assert has_wrapper(env, PassiveEnvChecker)

        env.reset()
        env.step(env.action_space.sample())
        env.render()

    assert all(
        any(
            re.match(ignore_warning, warning.message.args[0])
            for ignore_warning in IGNORE_WARNINGS
        )
        for warning in warnings
    ), [warning.message.args[0] for warning in warnings]

    env.close()
