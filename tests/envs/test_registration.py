# -*- coding: utf-8 -*-
import gym
from gym import error, envs
from gym.envs import registration
from gym.envs.classic_control import cartpole
from gym.envs.registration import EnvSpec, EnvSpecTree


class ArgumentEnv(gym.Env):
    def __init__(self, arg1, arg2, arg3):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3


gym.register(
    id="test.ArgumentEnv-v0",
    entry_point="tests.envs.test_registration:ArgumentEnv",
    kwargs={
        "arg1": "arg1",
        "arg2": "arg2",
    },
)


def test_make():
    env = envs.make("CartPole-v0")
    assert env.spec.id == "CartPole-v0"
    assert isinstance(env.unwrapped, cartpole.CartPoleEnv)


def test_make_with_kwargs():
    env = envs.make("test.ArgumentEnv-v0", arg2="override_arg2", arg3="override_arg3")
    assert env.spec.id == "test.ArgumentEnv-v0"
    assert isinstance(env.unwrapped, ArgumentEnv)
    assert env.arg1 == "arg1"
    assert env.arg2 == "override_arg2"
    assert env.arg3 == "override_arg3"


def test_make_deprecated():
    try:
        envs.make("Humanoid-v0")
    except error.Error:
        pass
    else:
        assert False


def test_spec():
    spec = envs.spec("CartPole-v0")
    assert spec.id == "CartPole-v0"


def test_spec_with_kwargs():
    map_name_value = "8x8"
    env = gym.make("FrozenLake-v1", map_name=map_name_value)
    assert env.spec._kwargs["map_name"] == map_name_value


def test_missing_lookup():
    registry = registration.EnvRegistry()
    registry.register(id="Test-v0", entry_point=None)
    registry.register(id="Test-v15", entry_point=None)
    registry.register(id="Test-v9", entry_point=None)
    registry.register(id="Other-v100", entry_point=None)
    try:
        registry.spec("Test-v1")  # must match an env name but not the version above
    except error.DeprecatedEnv:
        pass
    else:
        assert False

    try:
        registry.spec("Test-v1000")
    except error.UnregisteredEnv:
        pass
    else:
        assert False

    try:
        registry.spec("Unknown-v1")
    except error.UnregisteredEnv:
        pass
    else:
        assert False


def test_malformed_lookup():
    registry = registration.EnvRegistry()
    try:
        registry.spec(u"“Breakout-v0”")
    except error.Error as e:
        assert "malformed environment ID" in "{}".format(
            e
        ), "Unexpected message: {}".format(e)
    else:
        assert False


def test_env_spec_tree():
    spec_tree = EnvSpecTree()

    # Add with namespace
    spec = EnvSpec("test/Test-v0")
    spec_tree["test/Test-v0"] = spec
    assert spec_tree.tree.keys() == {"test"}
    assert spec_tree.tree["test"].keys() == {"Test"}
    assert spec_tree.tree["test"]["Test"].keys() == {"0"}
    assert spec_tree.tree["test"]["Test"]["0"] == spec
    assert spec_tree["test/Test-v0"] == spec

    # Add without namespace
    spec = EnvSpec("Test-v0")
    spec_tree["Test-v0"] = spec
    assert spec_tree.tree.keys() == {"test", None}
    assert spec_tree.tree[None].keys() == {"Test"}
    assert spec_tree.tree[None]["Test"].keys() == {"0"}
    assert spec_tree.tree[None]["Test"]["0"] == spec

    # Delete last version deletes entire subtree
    del spec_tree["test/Test-v0"]
    assert spec_tree.tree.keys() == {None}

    # Append second version for same name
    spec_tree["Test-v1"] = EnvSpec("Test-v1")
    assert spec_tree.tree.keys() == {None}
    assert spec_tree.tree[None].keys() == {"Test"}
    assert spec_tree.tree[None]["Test"].keys() == {"0", "1"}

    # Deleting one version leaves other
    del spec_tree["Test-v0"]
    assert spec_tree.tree.keys() == {None}
    assert spec_tree.tree[None].keys() == {"Test"}
    assert spec_tree.tree[None]["Test"].keys() == {"1"}
