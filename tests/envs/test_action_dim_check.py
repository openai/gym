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
    # Car racing currently allow `out-of-bound` actions
    if "CarRacing" in env.spec.id:
        return

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
    """Environments with Box actions spaces perform clipping inside `step`.
    The expected behaviour is that an action `out-of-bound` has the same effect
    of an action with value exactly at the upper (or lower) bound.
    """
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
            upper_bounds[i] += np.cast[dtype](100)
            oob_obs, _, _, _ = oob_env.step(upper_bounds)

            assert np.alltrue(obs == oob_obs)
            return

        elif is_lower_bound:
            obs, _, _, _ = env.step(lower_bounds)
            lower_bounds[i] -= np.cast[dtype](100)
            oob_obs, _, _, _ = oob_env.step(lower_bounds)

            assert np.alltrue(obs == oob_obs)
            return
