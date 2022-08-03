"""Test suite for TimeAwareobservationV0."""
from collections import OrderedDict

import numpy as np
import pytest

try:
    from gym.wrappers import TimeAwareObservationV0
except ImportError:
    pytest.skip(allow_module_level=True)

import gym
from gym.spaces import Box, Dict

NUM_STEPS = 20
NUM_ENVS = 3
SEED = 0


@pytest.mark.parametrize(
    "env",
    [
        gym.make("CartPole-v1", disable_env_checker=True, new_step_api=True),
        gym.make("CarRacing-v2", disable_env_checker=True, new_step_api=True),
        gym.vector.make(
            "CartPole-v1",
            disable_env_checker=True,
            new_step_api=True,
            num_envs=NUM_ENVS,
        ),
        gym.vector.make(
            "CarRacing-v2",
            disable_env_checker=True,
            new_step_api=True,
            num_envs=NUM_ENVS,
        ),
    ],
)
def test_time_aware_observation_creation(env):
    """Test TimeAwareObservationV0 wrapper creation.

    This test checks if wrapped env with TimeAwareObservationV0
    is correctly created.
    """
    wrapped_env = TimeAwareObservationV0(env)
    obs = wrapped_env.reset()

    assert isinstance(wrapped_env.observation_space, Dict)
    assert isinstance(obs, OrderedDict)
    assert np.all(obs["time"] == 0)
    assert env.observation_space == wrapped_env.observation_space["obs"]


@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize(
    "env",
    [
        gym.make("CartPole-v1", disable_env_checker=True, new_step_api=True),
        gym.make(
            "CarRacing-v2",
            disable_env_checker=True,
            new_step_api=True,
            continuous=False,
        ),
    ],
)
def test_time_aware_observation_step(env, flatten):
    """Test TimeAwareObservationV0 step.

    This test checks if wrapped env with TimeAwareObservationV0
    steps correctly.
    """
    env.action_space.seed(SEED)

    wrapped_env = TimeAwareObservationV0(env, flatten=flatten)
    wrapped_env.reset(seed=SEED)

    for timestep in range(1, NUM_STEPS):
        action = env.action_space.sample()
        observation, _, terminated, _, _ = wrapped_env.step(action)

        if flatten:
            assert observation[-1] == timestep
        else:
            assert observation["time"] == timestep

        if terminated:
            break


@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize(
    "env",
    [
        gym.vector.make(
            "CartPole-v1",
            disable_env_checker=True,
            new_step_api=True,
            num_envs=NUM_ENVS,
        ),
        gym.vector.make(
            "CarRacing-v2",
            disable_env_checker=True,
            new_step_api=True,
            num_envs=NUM_ENVS,
        ),
    ],
)
def test_time_aware_observation_step_within_vector(env, flatten):
    """Test TimeAwareObservationV0 step in vectorized environment.

    This tests checks if wrapped env with TimeAwareObservationV0
    steps correctly in vectorized environment.

    When a the i-th environment call `reset` on termination,
    the i-th `time` observation should also reset.
    """
    env.action_space.seed(SEED)

    wrapped_env = TimeAwareObservationV0(env, flatten=flatten)
    wrapped_env.reset(seed=SEED)

    terminated = np.zeros(NUM_ENVS, dtype=bool)
    for timestep in range(1, NUM_STEPS):
        action = env.action_space.sample()
        observation, _, terminated, _, _ = wrapped_env.step(action)

        if flatten:
            assert np.all(observation[-wrapped_env.num_envs :] == timestep)
        else:
            assert np.all(observation["time"] == timestep)

        if any(terminated):
            break

    action = env.action_space.sample()
    observation, _, _, _, _ = wrapped_env.step(action)

    if flatten:
        assert np.all(observation[-wrapped_env.num_envs :][np.where(terminated)] == 1)
        assert np.all(observation[-wrapped_env.num_envs :][np.where(~terminated)] != 1)
    else:
        assert np.all(observation["time"][np.where(terminated)] == 1)
        assert np.all(observation["time"][np.where(~terminated)] != 1)


@pytest.mark.parametrize(
    "env",
    [
        gym.make("CartPole-v1", disable_env_checker=True, new_step_api=True),
        gym.make("CarRacing-v2", disable_env_checker=True, new_step_api=True),
        gym.vector.make(
            "CartPole-v1",
            disable_env_checker=True,
            new_step_api=True,
            num_envs=NUM_ENVS,
        ),
        gym.vector.make(
            "CarRacing-v2",
            disable_env_checker=True,
            new_step_api=True,
            num_envs=NUM_ENVS,
        ),
    ],
)
def test_time_aware_observation_creation_flatten(env):
    """Test TimeAwareObservationV0 wrapper creation with `flatten=True`.

    This test checks if wrapped env with TimeAwareObservationV0
    is correctly created when the `flatten` parameter is set to `True`.

    When flattened, the observation space should be a 1 dimension `Box`
    with time appended to the end.
    In vectorized environments there will be `n` times observation at
    the end, where `n` is the number of environments.
    """
    wrapped_env = TimeAwareObservationV0(env, flatten=True)
    obs = wrapped_env.reset()

    assert isinstance(wrapped_env.observation_space, Box)
    assert isinstance(obs, np.ndarray)
    assert np.all(obs[-wrapped_env.num_envs :] == 0)
    assert env.observation_space == wrapped_env.time_aware_observation_space["obs"]
