"""Tests that gym.spec works as expected."""

import re

import pytest

import gym


def test_spec():
    spec = gym.spec("CartPole-v1")
    assert spec.id == "CartPole-v1"
    assert spec is gym.envs.registry["CartPole-v1"]


def test_spec_kwargs():
    map_name_value = "8x8"
    env = gym.make("FrozenLake-v1", map_name=map_name_value)
    assert env.spec.kwargs["map_name"] == map_name_value


def test_spec_missing_lookup():
    gym.register(id="Test1-v0", entry_point=None)
    gym.register(id="Test1-v15", entry_point=None)
    gym.register(id="Test1-v9", entry_point=None)
    gym.register(id="Other1-v100", entry_point=None)

    with pytest.raises(
        gym.error.DeprecatedEnv,
        match=re.escape(
            "Environment version v1 for `Test1` is deprecated. Please use `Test1-v15` instead."
        ),
    ):
        gym.spec("Test1-v1")

    with pytest.raises(
        gym.error.UnregisteredEnv,
        match=re.escape(
            "Environment version `v1000` for environment `Test1` doesn't exist. It provides versioned environments: [ `v0`, `v9`, `v15` ]."
        ),
    ):
        gym.spec("Test1-v1000")

    with pytest.raises(
        gym.error.UnregisteredEnv,
        match=re.escape("Environment Unknown1 doesn't exist. "),
    ):
        gym.spec("Unknown1-v1")


def test_spec_malformed_lookup():
    with pytest.raises(
        gym.error.Error,
        match=f'^{re.escape("Malformed environment ID: “Breakout-v0”.(Currently all IDs must be of the form [namespace/](env-name)-v(version). (namespace is optional))")}$',
    ):
        gym.spec("“Breakout-v0”")


def test_spec_versioned_lookups():
    gym.register("test/Test2-v5")

    with pytest.raises(
        gym.error.VersionNotFound,
        match=re.escape(
            "Environment version `v9` for environment `test/Test2` doesn't exist. It provides versioned environments: [ `v5` ]."
        ),
    ):
        gym.spec("test/Test2-v9")

    with pytest.raises(
        gym.error.DeprecatedEnv,
        match=re.escape(
            "Environment version v4 for `test/Test2` is deprecated. Please use `test/Test2-v5` instead."
        ),
    ):
        gym.spec("test/Test2-v4")

    assert gym.spec("test/Test2-v5") is not None


def test_spec_default_lookups():
    gym.register("test/Test3")

    with pytest.raises(
        gym.error.DeprecatedEnv,
        match=re.escape(
            "Environment version `v0` for environment `test/Test3` doesn't exist. It provides the default version test/Test3`."
        ),
    ):
        gym.spec("test/Test3-v0")

    assert gym.spec("test/Test3") is not None
