"""Tests that `gym.register` works as expected."""
import re
from typing import Optional

import pytest

import gym


@pytest.fixture(scope="function")
def register_testing_envs():
    """Registers testing environments."""
    namespace = "MyAwesomeNamespace"
    versioned_name = "MyAwesomeVersionedEnv"
    unversioned_name = "MyAwesomeUnversionedEnv"
    versions = [1, 3, 5]
    for version in versions:
        env_id = f"{namespace}/{versioned_name}-v{version}"
        gym.register(
            id=env_id,
            entry_point="tests.envs.utils_envs:ArgumentEnv",
            kwargs={
                "arg1": "arg1",
                "arg2": "arg2",
                "arg3": "arg3",
            },
        )
    gym.register(
        id=f"{namespace}/{unversioned_name}",
        entry_point="tests.env.utils_envs:ArgumentEnv",
        kwargs={
            "arg1": "arg1",
            "arg2": "arg2",
            "arg3": "arg3",
        },
    )

    yield

    for version in versions:
        env_id = f"{namespace}/{versioned_name}-v{version}"
        del gym.envs.registry[env_id]
    del gym.envs.registry[f"{namespace}/{unversioned_name}"]


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
def test_register(
    env_id: str, namespace: Optional[str], name: str, version: Optional[int]
):
    gym.register(env_id, "no-entry-point")
    assert gym.spec(env_id).id == env_id

    full_name = f"{name}"
    if namespace:
        full_name = f"{namespace}/{full_name}"
    if version is not None:
        full_name = f"{full_name}-v{version}"

    assert full_name in gym.envs.registry.keys()

    del gym.envs.registry[env_id]


@pytest.mark.parametrize(
    "env_id",
    [
        "“Breakout-v0”",
        "MyNotSoAwesomeEnv-vNone\n",
        "MyNamespace///MyNotSoAwesomeEnv-vNone",
    ],
)
def test_register_error(env_id):
    with pytest.raises(gym.error.Error, match=f"^Malformed environment ID: {env_id}"):
        gym.register(env_id, "no-entry-point")


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
def test_env_suggestions(register_testing_envs, env_id_input, env_id_suggested):
    with pytest.raises(
        gym.error.UnregisteredEnv, match=f"Did you mean: `{env_id_suggested}`?"
    ):
        gym.make(env_id_input, disable_env_checker=True)


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
    register_testing_envs, env_id_input, suggested_versions, default_version
):
    if default_version:
        with pytest.raises(
            gym.error.DeprecatedEnv,
            match="It provides the default version",  # env name,
        ):
            gym.make(env_id_input, disable_env_checker=True)
    else:
        with pytest.raises(
            gym.error.UnregisteredEnv,
            match=f"It provides versioned environments: \\[ {suggested_versions} \\]",
        ):
            gym.make(env_id_input, disable_env_checker=True)


def test_register_versioned_unversioned():
    # Register versioned then unversioned
    versioned_env = "Test/MyEnv-v0"
    gym.register(versioned_env, "no-entry-point")
    assert gym.envs.spec(versioned_env).id == versioned_env

    unversioned_env = "Test/MyEnv"
    with pytest.raises(
        gym.error.RegistrationError,
        match=re.escape(
            "Can't register the unversioned environment `Test/MyEnv` when the versioned environment `Test/MyEnv-v0` of the same name already exists"
        ),
    ):
        gym.register(unversioned_env, "no-entry-point")

    # Clean everything
    del gym.envs.registry[versioned_env]

    # Register unversioned then versioned
    gym.register(unversioned_env, "no-entry-point")
    assert gym.envs.spec(unversioned_env).id == unversioned_env
    with pytest.raises(
        gym.error.RegistrationError,
        match=re.escape(
            "Can't register the versioned environment `Test/MyEnv-v0` when the unversioned environment `Test/MyEnv` of the same name already exists."
        ),
    ):
        gym.register(versioned_env, "no-entry-point")

    # Clean everything
    del gym.envs.registry[unversioned_env]


def test_make_latest_versioned_env(register_testing_envs):
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Using the latest versioned environment `MyAwesomeNamespace/MyAwesomeVersionedEnv-v5` instead of the unversioned environment `MyAwesomeNamespace/MyAwesomeVersionedEnv`."
        ),
    ):
        env = gym.make(
            "MyAwesomeNamespace/MyAwesomeVersionedEnv", disable_env_checker=True
        )
    assert env.spec.id == "MyAwesomeNamespace/MyAwesomeVersionedEnv-v5"


def test_namespace():
    # Check if the namespace context manager works
    with gym.envs.registration.namespace("MyDefaultNamespace"):
        gym.register("MyDefaultEnvironment-v0", "no-entry-point")
    gym.register("MyDefaultEnvironment-v1", "no-entry-point")
    assert "MyDefaultNamespace/MyDefaultEnvironment-v0" in gym.envs.registry
    assert "MyDefaultEnvironment-v1" in gym.envs.registry

    del gym.envs.registry["MyDefaultNamespace/MyDefaultEnvironment-v0"]
    del gym.envs.registry["MyDefaultEnvironment-v1"]
