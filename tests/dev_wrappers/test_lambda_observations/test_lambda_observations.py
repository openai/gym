import numpy as np
import pytest

import gym
from gym.spaces import Box, Dict
from tests.dev_wrappers.mock_data import DISCRETE_ACTION, NUM_ENVS, SEED
from tests.dev_wrappers.utils import TestingEnv

try:
    from gym.wrappers import lambda_observations_v0
except ImportError:
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize(
    ("env", "func", "args"), [(gym.make("CartPole-v1", new_step_api=True), lambda obs, arg: obs * arg, 2)]
)
def test_lambda_observation_v0(env, func, args):
    """Test correct function is applied to observation."""
    env.reset(seed=SEED)
    obs, _, _, _, _ = env.step(DISCRETE_ACTION)

    wrapped_env = lambda_observations_v0(env, func, args)
    wrapped_env.reset(seed=SEED)
    lambda_obs, _, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    assert np.alltrue(lambda_obs == obs * args)


@pytest.mark.parametrize(
    ("env", "func", "args"),
    [
        (
            gym.vector.make("CartPole-v1", num_envs=NUM_ENVS, new_step_api=True),
            lambda obs, arg: obs * arg,
            2,
        ),
        (
            gym.vector.make("CartPole-v1", num_envs=NUM_ENVS, new_step_api=True),
            lambda obs, arg: obs * arg,
            np.array([[1], [-1], [0]]),  # shape (3,1) because NUM_ENVS = 3
        ),
    ],
)
def test_lambda_observation_v0_vector_env(env, func, args):
    """Test correct function is applied to observation in vector environment."""
    env.reset(seed=SEED)
    obs, _, _, _, _ = env.step([DISCRETE_ACTION for _ in range(NUM_ENVS)])

    wrapped_env = lambda_observations_v0(env, func, args)
    wrapped_env.reset(seed=SEED)
    lambda_obs, _, _, _, _ = wrapped_env.step([DISCRETE_ACTION for _ in range(NUM_ENVS)])

    assert np.alltrue(lambda_obs == obs * args)


@pytest.mark.parametrize(
    ("env", "func", "args"),
    [
        (
            TestingEnv(
                observation_space=Dict(
                    left_arm=Box(-5, 5, (1,)), right_arm=Box(-5, 5, (1,))
                )
            ),
            lambda obs, arg: obs * arg,
            {"left_arm": 0, "right_arm": 0},
        ),
    ],
)
def test_composite_space_lambda_observation_v0(env, func, args):
    """Test correct function is applied to observation.

    Test if function is applied correctly to composite action space.
    """
    env.reset(seed=SEED)
    obs, _, _, _ = env.step(DISCRETE_ACTION)

    wrapped_env = lambda_observations_v0(env, func, args)
    wrapped_env.reset(seed=SEED)
    lambda_obs, _, _, _ = wrapped_env.step(DISCRETE_ACTION)

    for arg, arg_value in args.items():
        assert np.alltrue(lambda_obs[arg] == obs[arg] * arg_value)
