from typing import List

import numpy as np
import pytest

from gym import envs
from gym.envs.registration import EnvSpec
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.space import Space
from tests.envs.spec_list import SKIP_MUJOCO_WARNING_MESSAGE, skip_mujoco, spec_list

ENVIRONMENT_IDS = ("HalfCheetah-v2",)


def make_envs_by_action_space_type(spec_list: List[EnvSpec], action_space: Space):
    """Make environments of specific action_space type.

    This function returns a filtered list of environment from the
    spec_list that matches the action_space type.

    Args:
        spec_list (list): list of registered environments' specification
        action_space (gym.spaces.Space): action_space type
    """
    filtered_envs = []
    for spec in spec_list:
        env = envs.make(spec.id)
        if isinstance(env.action_space, action_space):
            filtered_envs.append(env)
    return filtered_envs


@pytest.mark.skipif(skip_mujoco, reason=SKIP_MUJOCO_WARNING_MESSAGE)
@pytest.mark.parametrize("environment_id", ENVIRONMENT_IDS)
def test_serialize_deserialize(environment_id):
    env = envs.make(environment_id)
    env.reset()

    with pytest.raises(ValueError, match="Action dimension mismatch"):
        env.step([0.1])

    with pytest.raises(ValueError, match="Action dimension mismatch"):
        env.step(0.1)


@pytest.mark.parametrize("env", make_envs_by_action_space_type(spec_list, Discrete))
def test_discrete_actions_out_of_bound(env):
    """Test out of bound actions in Discrete action_space.

    In discrete action_space environments, `out-of-bound`
    actions are not allowed and should raise an exception.

    Args:
        env (gym.Env): the gym environment
    """
    env.reset()

    action_space = env.action_space
    upper_bound = action_space.start + action_space.n - 1

    with pytest.raises(Exception):
        env.step(upper_bound + 1)


@pytest.mark.parametrize(
    ("env", "seed"),
    [(env, 42) for env in make_envs_by_action_space_type(spec_list, Box)],
)
def test_box_actions_out_of_bound(env, seed):
    """Test out of bound actions in Box action_space.

    Environments with Box actions spaces perform clipping inside `step`.
    The expected behaviour is that an action `out-of-bound` has the same effect
    of an action with value exactly at the upper (or lower) bound.

    Args:
        env (gym.Env): the gym environment
        seed (int): seed value for determinism
    """
    OOB_VALUE = 100

    env.reset(seed=seed)

    oob_env = envs.make(env.spec.id)
    oob_env.reset(seed=seed)

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
