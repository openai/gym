"""Test lambda reward wrapper."""
import pytest

try:
    from gym.wrappers import clip_rewards_v0, lambda_reward_v0
except ImportError:
    pytest.skip(allow_module_level=True)

import jax.numpy as jnp
import numpy as np

import gym
from gym.error import InvalidBound

ENV_ID = "CartPole-v1"
DISCRETE_ACTION = 0
NUM_ENVS = 3
SEED = 0


@pytest.mark.parametrize(
    ("reward_fn", "expected_reward"),
    [(lambda r: 2 * r + 1, 3)],
)
def test_lambda_reward(reward_fn, expected_reward):
    """Test lambda reward.

    Tests if function is correctly applied
    to reward.
    """
    env = gym.make(ENV_ID)
    env = lambda_reward_v0(env, reward_fn)
    env.reset(seed=SEED)

    _, rew, _, _ = env.step(DISCRETE_ACTION)

    assert rew == expected_reward


@pytest.mark.parametrize(
    (
        "reward_fn",
        "expected_reward",
    ),
    [(lambda r: 2 * r + 1, 3)],
)
def test_lambda_reward_within_vector(reward_fn, expected_reward):
    """Test lambda reward in vectorized environment.

    Tests if function is correctly applied
    to reward in a vectorized environment.
    """
    actions = [DISCRETE_ACTION for _ in range(NUM_ENVS)]
    env = gym.vector.make(ENV_ID, num_envs=NUM_ENVS)
    env = lambda_reward_v0(env, reward_fn)
    env.reset(seed=SEED)

    _, rew, _, _ = env.step(actions)

    assert np.alltrue(rew == expected_reward)


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound", "expected_reward"),
    [(None, 0.5, 0.5), (0, None, 1), (0, 0.5, 0.5)],
)
def test_clip_reward(lower_bound, upper_bound, expected_reward):
    """Test reward clipping.

    Test if reward is correctly clipped
    accordingly to the input args.
    """
    env = gym.make(ENV_ID)
    env = clip_rewards_v0(env, lower_bound, upper_bound)
    env.reset(seed=SEED)
    _, rew, _, _ = env.step(DISCRETE_ACTION)

    assert rew == expected_reward


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound", "expected_reward"),
    [(None, 0.5, 0.5), (0, None, 1), (0, 0.5, 0.5)],
)
def test_clip_reward_within_vector(lower_bound, upper_bound, expected_reward):
    """Test reward clipping in vectorized environment.

    Test if reward is correctly clipped
    accordingly to the input args in a vectorized environment.
    """
    actions = [DISCRETE_ACTION for _ in range(NUM_ENVS)]

    env = gym.vector.make(ENV_ID, num_envs=NUM_ENVS)
    env = clip_rewards_v0(env, lower_bound, upper_bound)
    env.reset(seed=SEED)

    _, rew, _, _ = env.step(actions)

    assert np.alltrue(rew == expected_reward)


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound"),
    [
        (None, None),
        (1, -1),
        (np.array([1, 1]), np.array([0, 0])),
        (jnp.array([1, 1]), jnp.array([0, 0])),
    ],
)
def test_clip_reward_incorrect_params(lower_bound, upper_bound):
    """Test reward clipping with incorrect params.

    Test whether passing wrong params to clip_rewards
    correctly raise an exception.

    clip_rewards should raise an exception if, both low and upper
    bound of reward are `None` or if upper bound is lower than lower bound.
    """
    env = gym.make(ENV_ID)

    with pytest.raises(InvalidBound):
        env = clip_rewards_v0(env, lower_bound, upper_bound)
