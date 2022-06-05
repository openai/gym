import numpy as np
import pytest

import gym
from gym import envs

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
        np.testing.assert_allclose(old_done, new_done, atol=EPS)

        for key in old_info:
            np.testing.assert_allclose(old_info[key], new_info[key], atol=EPS)

        if old_done:
            break


MUJOCO_V2_V3_ENVS = [
    spec.name
    for spec in gym.envs.registry.values()
    if "mujoco" in spec.entry_point
    and spec.version == 2
    and f"{spec.name}-v3" in gym.envs.registry
]


@pytest.mark.parametrize("env_name", MUJOCO_V2_V3_ENVS)
def test_mujoco_v2_to_v3_conversion(env_name: str):
    verify_environments_match(f"{env_name}-v2", f"{env_name}-v3")


@pytest.mark.parametrize("env_name", MUJOCO_V2_V3_ENVS)
def test_mujoco_incompatible_v3_to_v2(env_name: str):
    # v3 mujoco environments have additional information so the info check will fail
    with pytest.raises(KeyError):
        verify_environments_match(f"{env_name}-v3", f"{env_name}-v2")


MUJOCO_V4_ENVS = [
    spec.name
    for spec in gym.envs.registry.values()
    if "mujoco" in spec.entry_point and spec.version == 4
]


@pytest.mark.parametrize("env_name", MUJOCO_V4_ENVS)
def test_mujoco_v4_envs(env_name):
    if f"{env_name}-v3" in gym.envs.registry:
        verify_environments_match(f"{env_name}-v3", f"{env_name}-v4")
    elif f"{env_name}-v2" in gym.envs.registry:
        verify_environments_match(f"{env_name}-v2", f"{env_name}-v4")
    else:
        raise Exception(f"Could not find v2 or v3 mujoco env for {env_name}")
