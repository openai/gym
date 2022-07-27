import numpy as np
import pytest

import gym
from gym import envs
from gym.envs.registration import EnvSpec
from tests.envs.utils import mujoco_testing_env_specs

EPS = 1e-6


def verify_environments_match(
    old_env_id: str, new_env_id: str, seed: int = 1, num_actions: int = 1000
):
    """Verifies with two environment ids (old and new) are identical in obs, reward and done
    (except info where all old info must be contained in new info)."""
    old_env = envs.make(old_env_id, disable_env_checker=True)
    new_env = envs.make(new_env_id, disable_env_checker=True)

    old_reset_obs = old_env.reset(seed=seed)
    new_reset_obs = new_env.reset(seed=seed)

    np.testing.assert_allclose(old_reset_obs, new_reset_obs)

    for i in range(num_actions):
        action = old_env.action_space.sample()
        old_obs, old_reward, old_done, old_info = old_env.step(action)
        new_obs, new_reward, new_done, new_info = new_env.step(action)

        np.testing.assert_allclose(old_obs, new_obs, atol=EPS)
        np.testing.assert_allclose(old_reward, new_reward, atol=EPS)
        np.testing.assert_equal(old_done, new_done)

        for key in old_info:
            np.testing.assert_allclose(old_info[key], new_info[key], atol=EPS)

        if old_done:
            break


EXCLUDE_POS_FROM_OBS = [
    "Ant",
    "HalfCheetah",
    "Hopper",
    "Humanoid",
    "Swimmer",
    "Walker2d",
]


@pytest.mark.parametrize(
    "env_spec",
    mujoco_testing_env_specs,
    ids=[env_spec.id for env_spec in mujoco_testing_env_specs],
)
def test_obs_space_mujoco_environments(env_spec: EnvSpec):
    """Check that the returned observations are contained in the observation space of the environment"""
    env = env_spec.make(disable_env_checker=True)
    reset_obs = env.reset()
    assert env.observation_space.contains(
        reset_obs
    ), f"Obseravtion returned by reset() of {env_spec.id} is not contained in the default observation space {env.observation_space}."

    action = env.action_space.sample()
    step_obs, _, _, _ = env.step(action)
    assert env.observation_space.contains(
        step_obs
    ), f"Obseravtion returned by step(action) of {env_spec.id} is not contained in the default observation space {env.observation_space}."

    if env_spec.name in EXCLUDE_POS_FROM_OBS and (
        env_spec.version == 4 or env_spec.version == 3
    ):
        env = env_spec.make(
            disable_env_checker=True, exclude_current_positions_from_observation=False
        )
        reset_obs = env.reset()
        assert env.observation_space.contains(
            reset_obs
        ), f"Obseravtion of {env_spec.id} is not contained in the default observation space {env.observation_space} when excluding current position from observation."

        step_obs, _, _, _ = env.step(action)
        assert env.observation_space.contains(
            step_obs
        ), f"Obseravtion returned by step(action) of {env_spec.id} is not contained in the default observation space {env.observation_space} when excluding current position from observation."

    # Ant-v4 has the option of including contact forces in the observation space with the use_contact_forces argument
    if env_spec.name == "Ant" and env_spec.version == 4:
        env = env_spec.make(disable_env_checker=True, use_contact_forces=True)
        reset_obs = env.reset()
        assert env.observation_space.contains(
            reset_obs
        ), f"Obseravtion of {env_spec.id} is not contained in the default observation space {env.observation_space} when using contact forces."

        step_obs, _, _, _ = env.step(action)
        assert env.observation_space.contains(
            step_obs
        ), f"Obseravtion returned by step(action) of {env_spec.id} is not contained in the default observation space {env.observation_space} when using contact forces."


MUJOCO_V2_V3_ENVS = [
    spec.name
    for spec in mujoco_testing_env_specs
    if spec.version == 2 and f"{spec.name}-v3" in gym.envs.registry
]


@pytest.mark.parametrize("env_name", MUJOCO_V2_V3_ENVS)
def test_mujoco_v2_to_v3_conversion(env_name: str):
    """Checks that all v2 mujoco environments are the same as v3 environments."""
    verify_environments_match(f"{env_name}-v2", f"{env_name}-v3")


@pytest.mark.parametrize("env_name", MUJOCO_V2_V3_ENVS)
def test_mujoco_incompatible_v3_to_v2(env_name: str):
    """Checks that the v3 environment are slightly different from v2, (v3 has additional info keys that v2 does not)."""
    with pytest.raises(KeyError):
        verify_environments_match(f"{env_name}-v3", f"{env_name}-v2")
