"""Tests that gym.make works as expected."""

import re
import warnings
from copy import deepcopy

import numpy as np
import pytest

import gym
from gym.envs.classic_control import cartpole
from gym.wrappers import AutoResetWrapper, HumanRendering, OrderEnforcing, TimeLimit
from gym.wrappers.env_checker import PassiveEnvChecker
from tests.envs.test_envs import PASSIVE_CHECK_IGNORE_WARNING
from tests.envs.utils import all_testing_env_specs
from tests.envs.utils_envs import ArgumentEnv, RegisterDuringMakeEnv
from tests.testing_env import GenericTestEnv, old_step_fn
from tests.wrappers.utils import has_wrapper

gym.register(
    "RegisterDuringMakeEnv-v0",
    entry_point="tests.envs.utils_envs:RegisterDuringMakeEnv",
)

gym.register(
    id="test.ArgumentEnv-v0",
    entry_point="tests.envs.utils_envs:ArgumentEnv",
    kwargs={
        "arg1": "arg1",
        "arg2": "arg2",
    },
)

gym.register(
    id="test/NoHuman-v0",
    entry_point="tests.envs.utils_envs:NoHuman",
)
gym.register(
    id="test/NoHumanOldAPI-v0",
    entry_point="tests.envs.utils_envs:NoHumanOldAPI",
)

gym.register(
    id="test/NoHumanNoRGB-v0",
    entry_point="tests.envs.utils_envs:NoHumanNoRGB",
)


def test_make():
    env = gym.make("CartPole-v1", disable_env_checker=True)
    assert env.spec.id == "CartPole-v1"
    assert isinstance(env.unwrapped, cartpole.CartPoleEnv)
    env.close()


def test_make_deprecated():
    with warnings.catch_warnings(record=True):
        with pytest.raises(
            gym.error.Error,
            match=re.escape(
                "Environment version v0 for `Humanoid` is deprecated. Please use `Humanoid-v4` instead."
            ),
        ):
            gym.make("Humanoid-v0", disable_env_checker=True)


def test_make_max_episode_steps():
    # Default, uses the spec's
    env = gym.make("CartPole-v1", disable_env_checker=True)
    assert has_wrapper(env, TimeLimit)
    assert (
        env.spec.max_episode_steps == gym.envs.registry["CartPole-v1"].max_episode_steps
    )
    env.close()

    # Custom max episode steps
    env = gym.make("CartPole-v1", max_episode_steps=100, disable_env_checker=True)
    assert has_wrapper(env, TimeLimit)
    assert env.spec.max_episode_steps == 100
    env.close()

    # Env spec has no max episode steps
    assert gym.spec("test.ArgumentEnv-v0").max_episode_steps is None
    env = gym.make(
        "test.ArgumentEnv-v0", arg1=None, arg2=None, arg3=None, disable_env_checker=True
    )
    assert has_wrapper(env, TimeLimit) is False
    env.close()


def test_gym_make_autoreset():
    """Tests that `gym.make` autoreset wrapper is applied only when `gym.make(..., autoreset=True)`."""
    env = gym.make("CartPole-v1", disable_env_checker=True)
    assert has_wrapper(env, AutoResetWrapper) is False
    env.close()

    env = gym.make("CartPole-v1", autoreset=False, disable_env_checker=True)
    assert has_wrapper(env, AutoResetWrapper) is False
    env.close()

    env = gym.make("CartPole-v1", autoreset=True)
    assert has_wrapper(env, AutoResetWrapper)
    env.close()


def test_make_disable_env_checker():
    """Tests that `gym.make` disable env checker is applied only when `gym.make(..., disable_env_checker=False)`."""
    spec = deepcopy(gym.spec("CartPole-v1"))

    # Test with spec disable env checker
    spec.disable_env_checker = False
    env = gym.make(spec)
    assert has_wrapper(env, PassiveEnvChecker)
    env.close()

    # Test with overwritten spec using make disable env checker
    assert spec.disable_env_checker is False
    env = gym.make(spec, disable_env_checker=True)
    assert has_wrapper(env, PassiveEnvChecker) is False
    env.close()

    # Test with spec enabled disable env checker
    spec.disable_env_checker = True
    env = gym.make(spec)
    assert has_wrapper(env, PassiveEnvChecker) is False
    env.close()

    # Test with overwritten spec using make disable env checker
    assert spec.disable_env_checker is True
    env = gym.make(spec, disable_env_checker=False)
    assert has_wrapper(env, PassiveEnvChecker)
    env.close()


def test_apply_api_compatibility():
    gym.register(
        "testing-old-env",
        lambda: GenericTestEnv(step_fn=old_step_fn),
        apply_api_compatibility=True,
        max_episode_steps=3,
    )
    env = gym.make("testing-old-env")

    env.reset()
    assert len(env.step(env.action_space.sample())) == 5
    env.step(env.action_space.sample())
    _, _, termination, truncation, _ = env.step(env.action_space.sample())
    assert termination is False and truncation is True

    gym.spec("testing-old-env").apply_api_compatibility = False
    env = gym.make("testing-old-env")
    # Cannot run reset and step as will not work

    env = gym.make("testing-old-env", apply_api_compatibility=True)

    env.reset()
    assert len(env.step(env.action_space.sample())) == 5
    env.step(env.action_space.sample())
    _, _, termination, truncation, _ = env.step(env.action_space.sample())
    assert termination is False and truncation is True

    gym.envs.registry.pop("testing-old-env")


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_passive_checker_wrapper_warnings(spec):
    with warnings.catch_warnings(record=True) as caught_warnings:
        env = gym.make(spec)  # disable_env_checker=False
        env.reset()
        env.step(env.action_space.sample())
        # todo, add check for render, bugged due to mujoco v2/3 and v4 envs

        env.close()

    for warning in caught_warnings:
        if warning.message.args[0] not in PASSIVE_CHECK_IGNORE_WARNING:
            raise gym.error.Error(f"Unexpected warning: {warning.message}")


def test_make_order_enforcing():
    """Checks that gym.make wrappers the environment with the OrderEnforcing wrapper."""
    assert all(spec.order_enforce is True for spec in all_testing_env_specs)

    env = gym.make("CartPole-v1", disable_env_checker=True)
    assert has_wrapper(env, OrderEnforcing)
    # We can assume that there all other specs will also have the order enforcing
    env.close()

    gym.register(
        id="test.OrderlessArgumentEnv-v0",
        entry_point="tests.envs.utils_envs:ArgumentEnv",
        order_enforce=False,
        kwargs={"arg1": None, "arg2": None, "arg3": None},
    )

    env = gym.make("test.OrderlessArgumentEnv-v0", disable_env_checker=True)
    assert has_wrapper(env, OrderEnforcing) is False
    env.close()


def test_make_render_mode():
    env = gym.make("CartPole-v1", disable_env_checker=True)
    assert env.render_mode is None
    env.close()

    # Make sure that render_mode is applied correctly
    env = gym.make(
        "CartPole-v1", render_mode="rgb_array_list", disable_env_checker=True
    )
    assert env.render_mode == "rgb_array_list"
    env.reset()
    renders = env.render()
    assert isinstance(
        renders, list
    )  # Make sure that the `render` method does what is supposed to
    assert isinstance(renders[0], np.ndarray)
    env.close()

    env = gym.make("CartPole-v1", render_mode=None, disable_env_checker=True)
    assert env.render_mode is None
    valid_render_modes = env.metadata["render_modes"]
    env.close()

    assert len(valid_render_modes) > 0
    with warnings.catch_warnings(record=True) as caught_warnings:
        env = gym.make(
            "CartPole-v1", render_mode=valid_render_modes[0], disable_env_checker=True
        )
        assert env.render_mode == valid_render_modes[0]
        env.close()

    for warning in caught_warnings:
        raise gym.error.Error(f"Unexpected warning: {warning.message}")

    # Make sure that native rendering is used when possible
    env = gym.make("CartPole-v1", render_mode="human", disable_env_checker=True)
    assert not has_wrapper(env, HumanRendering)  # Should use native human-rendering
    assert env.render_mode == "human"
    env.close()

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "You are trying to use 'human' rendering for an environment that doesn't natively support it. The HumanRendering wrapper is being applied to your environment."
        ),
    ):
        # Make sure that `HumanRendering` is applied here
        env = gym.make(
            "test/NoHuman-v0", render_mode="human", disable_env_checker=True
        )  # This environment doesn't use native rendering
        assert has_wrapper(env, HumanRendering)
        assert env.render_mode == "human"
        env.close()

    with pytest.raises(
        TypeError, match=re.escape("got an unexpected keyword argument 'render_mode'")
    ):
        gym.make(
            "test/NoHumanOldAPI-v0",
            render_mode="rgb_array_list",
            disable_env_checker=True,
        )

    # Make sure that an additional error is thrown a user tries to use the wrapper on an environment with old API
    with warnings.catch_warnings(record=True):
        with pytest.raises(
            gym.error.Error,
            match=re.escape(
                "You passed render_mode='human' although test/NoHumanOldAPI-v0 doesn't implement human-rendering natively."
            ),
        ):
            gym.make(
                "test/NoHumanOldAPI-v0", render_mode="human", disable_env_checker=True
            )

    # This test ensures that the additional exception "Gym tried to apply the HumanRendering wrapper but it looks like
    # your environment is using the old rendering API" is *not* triggered by a TypeError that originate from
    # a keyword that is not `render_mode`
    with pytest.raises(
        TypeError,
        match=re.escape("got an unexpected keyword argument 'render'"),
    ):
        gym.make("CarRacing-v2", render="human")


def test_make_kwargs():
    env = gym.make(
        "test.ArgumentEnv-v0",
        arg2="override_arg2",
        arg3="override_arg3",
        disable_env_checker=True,
    )
    assert env.spec.id == "test.ArgumentEnv-v0"
    assert isinstance(env.unwrapped, ArgumentEnv)
    assert env.arg1 == "arg1"
    assert env.arg2 == "override_arg2"
    assert env.arg3 == "override_arg3"
    env.close()


def test_import_module_during_make():
    # Test custom environment which is registered at make
    env = gym.make(
        "tests.envs.utils:RegisterDuringMakeEnv-v0",
        disable_env_checker=True,
    )
    assert isinstance(env.unwrapped, RegisterDuringMakeEnv)
    env.close()
