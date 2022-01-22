# -*- coding: utf-8 -*-
import pytest

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


@pytest.fixture(scope="function")
def register_some_envs():
    namespace = "MyAwesomeNamespace"
    versioned_name = "MyAwesomeVersionedEnv"
    unversioned_name = "MyAwesomeUnversionedEnv"
    versions = [1, 3, 5]
    for version in versions:
        env_id = f"{namespace}/{versioned_name}-v{version}"
        gym.register(
            id=env_id,
            entry_point="tests.envs.test_registration:ArgumentEnv",
            kwargs={
                "arg1": "arg1",
                "arg2": "arg2",
                "arg3": "arg3",
            },
        )
    gym.register(
        id=f"{namespace}/{unversioned_name}",
        entry_point="tests.env.test_registration:ArgumentEnv",
        kwargs={
            "arg1": "arg1",
            "arg2": "arg2",
            "arg3": "arg3",
        },
    )

    yield

    for version in versions:
        env_id = f"{namespace}/{versioned_name}-v{version}"
        del gym.envs.registry.env_specs[env_id]
    del gym.envs.registry.env_specs[f"{namespace}/{unversioned_name}"]


def test_make():
    env = envs.make("CartPole-v0")
    assert env.spec.id == "CartPole-v0"
    assert isinstance(env.unwrapped, cartpole.CartPoleEnv)


@pytest.mark.parametrize(
    "env_id, namespace, name, version",
    [
        (
            "MyAwesomeNamespace/MyAwesomeEnv-v0",
            "MyAwesomeNamespace",
            "MyAwesomeEnv",
            0,
        ),
        ("MyAwesomeEnv-v0", None, "MyAwesomeEnv", 0),
        ("MyAwesomeEnv", None, "MyAwesomeEnv", None),
        ("MyAwesomeEnv-vfinal-v0", None, "MyAwesomeEnv-vfinal", 0),
        ("MyAwesomeEnv-vfinal", None, "MyAwesomeEnv-vfinal", None),
        ("MyAwesomeEnv--", None, "MyAwesomeEnv--", None),
        ("MyAwesomeEnv-v", None, "MyAwesomeEnv-v", None),
    ],
)
def test_register(env_id, namespace, name, version):
    envs.register(env_id)
    assert gym.envs.spec(env_id).id == env_id
    assert version in gym.envs.registry.env_specs.tree[namespace][name].keys()
    del gym.envs.registry.env_specs[env_id]


@pytest.mark.parametrize(
    "env_id",
    [
        "“Breakout-v0”",
        "MyNotSoAwesomeEnv-vNone\n",
        "MyNamespace///MyNotSoAwesomeEnv-vNone",
    ],
)
def test_register_error(env_id):
    with pytest.raises(error.Error, match="Malformed environment ID"):
        envs.register(env_id)


@pytest.mark.parametrize(
    "env_id_input, env_id_suggested",
    [
        ("cartpole-v1", "CartPole"),
        ("blackjack-v1", "Blackjack"),
        ("Blackjock-v1", "Blackjack"),
        ("mountaincarcontinuous-v0", "MountainCarContinuous"),
        ("taxi-v3", "Taxi"),
        ("taxi-v30", "Taxi"),
        ("MyAwesomeNamspce/MyAwesomeVersionedEnv-v1", "MyAwesomeNamespace"),
        ("MyAwesomeNamspce/MyAwesomeUnversionedEnv", "MyAwesomeNamespace"),
        ("MyAwesomeNamespace/MyAwesomeUnversioneEnv", "MyAwesomeUnversionedEnv"),
        ("MyAwesomeNamespace/MyAwesomeVersioneEnv", "MyAwesomeVersionedEnv"),
    ],
)
def test_env_suggestions(register_some_envs, env_id_input, env_id_suggested):
    with pytest.raises(
        error.UnregisteredEnv, match=f"Did you mean: `{env_id_suggested}` ?"
    ):
        envs.make(env_id_input)


@pytest.mark.parametrize(
    "env_id_input, suggested_versions, default_version",
    [
        ("CartPole-v12", "`v0`, `v1`", False),
        ("Blackjack-v10", "`v1`", False),
        ("MountainCarContinuous-v100", "`v0`", False),
        ("Taxi-v30", "`v3`", False),
        ("MyAwesomeNamespace/MyAwesomeVersionedEnv-v6", "`v1`, `v3`, `v5`", False),
        ("MyAwesomeNamespace/MyAwesomeUnversionedEnv-v6", "", True),
    ],
)
def test_env_version_suggestions(
    register_some_envs, env_id_input, suggested_versions, default_version
):
    if default_version:
        match_str = "provides the default version"
        with pytest.raises(
            error.DeprecatedEnv,
            match=match_str,
        ):
            envs.make(env_id_input)
    else:
        match_str = f"versioned environments: \\[ {suggested_versions} \\]"
        with pytest.raises(
            error.UnregisteredEnv,
            match=match_str,
        ):
            envs.make(env_id_input)


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
    assert env.spec.kwargs["map_name"] == map_name_value


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
        registry.spec("“Breakout-v0”")
    except error.Error as e:
        assert "Malformed environment ID" in f"{e}", f"Unexpected message: {e}"
    else:
        assert False


def test_versioned_lookups():
    registry = registration.EnvRegistry()
    registry.register("test/Test-v5")

    with pytest.raises(error.VersionNotFound):
        registry.spec("test/Test-v9")

    with pytest.raises(error.DeprecatedEnv):
        registry.spec("test/Test-v4")

    assert registry.spec("test/Test-v5")


def test_default_lookups():
    registry = registration.EnvRegistry()
    registry.register("test/Test")

    with pytest.raises(error.DeprecatedEnv):
        registry.spec("test/Test-v0")

    # Lookup default
    registry.spec("test/Test")


def test_env_spec_tree():
    spec_tree = EnvSpecTree()

    # Add with namespace
    spec = EnvSpec("test/Test-v0")
    spec_tree["test/Test-v0"] = spec
    assert spec_tree.tree.keys() == {"test"}
    assert spec_tree.tree["test"].keys() == {"Test"}
    assert spec_tree.tree["test"]["Test"].keys() == {0}
    assert spec_tree.tree["test"]["Test"][0] == spec
    assert spec_tree["test/Test-v0"] == spec

    # Add without namespace
    spec = EnvSpec("Test-v0")
    spec_tree["Test-v0"] = spec
    assert spec_tree.tree.keys() == {"test", None}
    assert spec_tree.tree[None].keys() == {"Test"}
    assert spec_tree.tree[None]["Test"].keys() == {0}
    assert spec_tree.tree[None]["Test"][0] == spec

    # Delete last version deletes entire subtree
    del spec_tree["test/Test-v0"]
    assert spec_tree.tree.keys() == {None}

    # Append second version for same name
    spec_tree["Test-v1"] = EnvSpec("Test-v1")
    assert spec_tree.tree.keys() == {None}
    assert spec_tree.tree[None].keys() == {"Test"}
    assert spec_tree.tree[None]["Test"].keys() == {0, 1}

    # Deleting one version leaves other
    del spec_tree["Test-v0"]
    assert spec_tree.tree.keys() == {None}
    assert spec_tree.tree[None].keys() == {"Test"}
    assert spec_tree.tree[None]["Test"].keys() == {1}

    # Add without version
    myenv = "MyAwesomeEnv"
    spec = EnvSpec(myenv)
    spec_tree[myenv] = spec
    assert spec_tree.tree.keys() == {None}
    assert myenv in spec_tree.tree[None].keys()
    assert spec_tree.tree[None][myenv].keys() == {None}
    assert spec_tree.tree[None][myenv][None] == spec
    assert spec_tree.__repr__() == "├──Test: [ v1 ]\n" + f"└──{myenv}: [  ]\n"


def test_register_versioned_unversioned():
    # Register versioned then unversioned
    versioned_env = "Test/MyEnv-v0"
    envs.register(versioned_env)
    assert gym.envs.spec(versioned_env).id == versioned_env
    unversioned_env = "Test/MyEnv"
    with pytest.raises(error.RegistrationError):
        envs.register(unversioned_env)

    # Clean everything
    del gym.envs.registry.env_specs[versioned_env]

    # Register unversioned then versioned
    with pytest.warns(UserWarning):
        envs.register(unversioned_env)
    assert gym.envs.spec(unversioned_env).id == unversioned_env
    with pytest.raises(error.RegistrationError):
        envs.register(versioned_env)

    # Clean everything
    envs_list = [versioned_env, unversioned_env]
    for env in envs_list:
        del gym.envs.registry.env_specs[env]


def test_return_latest_versioned_env(register_some_envs):
    with pytest.warns(UserWarning):
        env = envs.make("MyAwesomeNamespace/MyAwesomeVersionedEnv")
    assert env.spec.id == "MyAwesomeNamespace/MyAwesomeVersionedEnv-v5"
