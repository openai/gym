import numpy as np
import pytest

import gym
from gym.envs.registration import EnvSpec
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from tests.envs.utils import all_testing_initialised_envs, mujoco_testing_env_specs


@pytest.mark.parametrize(
    "env_spec",
    mujoco_testing_env_specs,
    ids=[env_spec.id for env_spec in mujoco_testing_env_specs],
)
def test_mujoco_action_dimensions(env_spec: EnvSpec):
    """Test that mujoco step actions if out of bounds then an error is raised."""
    env = env_spec.make(disable_env_checker=True)
    env.reset()

    with pytest.raises(ValueError, match="Action dimension mismatch"):
        env.step([0.1])

    with pytest.raises(ValueError, match="Action dimension mismatch"):
        env.step(0.1)


@pytest.mark.parametrize(
    "env",
    filter(
        lambda env: isinstance(env.action_space, Discrete), all_testing_initialised_envs
    ),
)
def test_discrete_actions_out_of_bound(env):
    """Test out of bound actions in Discrete action_space.

    In discrete action_space environments, `out-of-bound`
    actions are not allowed and should raise an exception.

    Args:
        env (gym.Env): the gym environment
    """
    assert isinstance(env.action_space, Discrete)
    upper_bound = env.action_space.start + env.action_space.n - 1

    env.reset()
    with pytest.raises(Exception):
        env.step(upper_bound + 1)


OOB_VALUE = 100


@pytest.mark.parametrize(
    "env", filter(lambda env: isinstance(env, Box), all_testing_initialised_envs)
)
def test_box_actions_out_of_bound(env: gym.Env):
    """Test out of bound actions in Box action_space.

    Environments with Box actions spaces perform clipping inside `step`.
    The expected behaviour is that an action `out-of-bound` has the same effect
    of an action with value exactly at the upper (or lower) bound.

    Args:
        env (gym.Env): the gym environment
    """
    env.reset(seed=42)

    oob_env = gym.make(env.spec.id)
    oob_env.reset(seed=42)

    assert isinstance(env.action_space, Box)
    dtype = env.action_space.dtype
    upper_bounds = env.action_space.high
    lower_bounds = env.action_space.low

    for i, (is_upper_bound, is_lower_bound) in enumerate(
        zip(env.action_space.bounded_above, env.action_space.bounded_below)
    ):
        if is_upper_bound:
            obs, _, _, _ = env.step(upper_bounds)
            oob_action = upper_bounds.copy()
            oob_action[i] += np.cast[dtype](OOB_VALUE)

            assert oob_action[i] > upper_bounds[i]
            oob_obs, _, _, _ = oob_env.step(oob_action)

            assert np.alltrue(obs == oob_obs)

        if is_lower_bound:
            obs, _, _, _ = env.step(lower_bounds)
            oob_action = lower_bounds.copy()
            oob_action[i] -= np.cast[dtype](OOB_VALUE)

            assert oob_action[i] < lower_bounds[i]
            oob_obs, _, _, _ = oob_env.step(oob_action)

            assert np.alltrue(obs == oob_obs)
