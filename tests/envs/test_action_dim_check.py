import numpy as np
import pytest

import gym
from gym import spaces
from gym.envs.registration import EnvSpec
from tests.envs.utils import all_testing_initialised_envs, mujoco_testing_env_specs


@pytest.mark.parametrize(
    "env_spec",
    mujoco_testing_env_specs,
    ids=[env_spec.id for env_spec in mujoco_testing_env_specs],
)
def test_mujoco_action_dimensions(env_spec: EnvSpec):
    """Test that for all mujoco environment, mis-dimensioned actions, an error is raised.

    Types of mis-dimensioned actions:
     * Too few actions
     * Too many actions
     * Too few dimensions
     * Too many dimensions
     * Incorrect shape
    """
    env = env_spec.make(disable_env_checker=True)
    env.reset()

    # Too few actions
    with pytest.raises(ValueError, match="Action dimension mismatch"):
        env.step(env.action_space.sample()[1:])

    # Too many actions
    with pytest.raises(ValueError, match="Action dimension mismatch"):
        env.step(np.append(env.action_space.sample(), 0))

    # Too few dimensions
    with pytest.raises(ValueError, match="Action dimension mismatch"):
        env.step(0.1)

    # Too many dimensions
    with pytest.raises(ValueError, match="Action dimension mismatch"):
        env.step(np.expand_dims(env.action_space.sample(), 0))

    # Incorrect shape
    with pytest.raises(ValueError, match="Action dimension mismatch"):
        env.step(np.expand_dims(env.action_space.sample(), 1))

    env.close()


DISCRETE_ENVS = list(
    filter(
        lambda env: isinstance(env.action_space, spaces.Discrete),
        all_testing_initialised_envs,
    )
)


@pytest.mark.parametrize(
    "env", DISCRETE_ENVS, ids=[env.spec.id for env in DISCRETE_ENVS]
)
def test_discrete_actions_out_of_bound(env: gym.Env):
    """Test out of bound actions in Discrete action_space.

    In discrete action_space environments, `out-of-bound`
    actions are not allowed and should raise an exception.

    Args:
        env (gym.Env): the gym environment
    """
    assert isinstance(env.action_space, spaces.Discrete)
    upper_bound = env.action_space.start + env.action_space.n - 1

    env.reset()
    with pytest.raises(Exception):
        env.step(upper_bound + 1)

    env.close()


BOX_ENVS = list(
    filter(
        lambda env: isinstance(env.action_space, spaces.Box),
        all_testing_initialised_envs,
    )
)
OOB_VALUE = 100


@pytest.mark.parametrize("env", BOX_ENVS, ids=[env.spec.id for env in BOX_ENVS])
def test_box_actions_out_of_bound(env: gym.Env):
    """Test out of bound actions in Box action_space.

    Environments with Box actions spaces perform clipping inside `step`.
    The expected behaviour is that an action `out-of-bound` has the same effect
    of an action with value exactly at the upper (or lower) bound.

    Args:
        env (gym.Env): the gym environment
    """
    env.reset(seed=42)

    oob_env = gym.make(env.spec.id, disable_env_checker=True)
    oob_env.reset(seed=42)

    assert isinstance(env.action_space, spaces.Box)
    dtype = env.action_space.dtype
    upper_bounds = env.action_space.high
    lower_bounds = env.action_space.low

    for i, (is_upper_bound, is_lower_bound) in enumerate(
        zip(env.action_space.bounded_above, env.action_space.bounded_below)
    ):
        if is_upper_bound:
            obs, _, _, _, _ = env.step(upper_bounds)
            oob_action = upper_bounds.copy()
            oob_action[i] += np.cast[dtype](OOB_VALUE)

            assert oob_action[i] > upper_bounds[i]
            oob_obs, _, _, _, _ = oob_env.step(oob_action)

            assert np.alltrue(obs == oob_obs)

        if is_lower_bound:
            obs, _, _, _, _ = env.step(
                lower_bounds
            )  # `env` is unwrapped, and in new step API
            oob_action = lower_bounds.copy()
            oob_action[i] -= np.cast[dtype](OOB_VALUE)

            assert oob_action[i] < lower_bounds[i]
            oob_obs, _, _, _, _ = oob_env.step(oob_action)

            assert np.alltrue(obs == oob_obs)

    env.close()
