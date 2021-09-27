import pytest
import numpy as np

from multiprocessing import TimeoutError
from gym.spaces import Box, Tuple
from gym.error import AlreadyPendingCallError, NoAsyncCallError, ClosedEnvironmentError
from tests.vector.utils import (
    CustomSpace,
    make_env,
    make_slow_env,
    make_custom_space_env,
)

from gym.vector.async_vector_env import AsyncVectorEnv


@pytest.mark.parametrize("shared_memory", [True, False])
def test_create_async_vector_env(shared_memory):
    env_fns = [make_env("CubeCrash-v0", i) for i in range(8)]
    try:
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    finally:
        env.close()

    assert env.num_envs == 8


@pytest.mark.parametrize("shared_memory", [True, False])
def test_reset_async_vector_env(shared_memory):
    env_fns = [make_env("CubeCrash-v0", i) for i in range(8)]
    try:
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
        observations = env.reset()
    finally:
        env.close()

    assert isinstance(env.observation_space, Box)
    assert isinstance(observations, np.ndarray)
    assert observations.dtype == env.observation_space.dtype
    assert observations.shape == (8,) + env.single_observation_space.shape
    assert observations.shape == env.observation_space.shape


@pytest.mark.parametrize("shared_memory", [True, False])
@pytest.mark.parametrize("use_single_action_space", [True, False])
def test_step_async_vector_env(shared_memory, use_single_action_space):
    env_fns = [make_env("CubeCrash-v0", i) for i in range(8)]
    try:
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
        observations = env.reset()
        if use_single_action_space:
            actions = [env.single_action_space.sample() for _ in range(8)]
        else:
            actions = env.action_space.sample()
        observations, rewards, dones, _ = env.step(actions)
    finally:
        env.close()

    assert isinstance(env.observation_space, Box)
    assert isinstance(observations, np.ndarray)
    assert observations.dtype == env.observation_space.dtype
    assert observations.shape == (8,) + env.single_observation_space.shape
    assert observations.shape == env.observation_space.shape

    assert isinstance(rewards, np.ndarray)
    assert isinstance(rewards[0], (float, np.floating))
    assert rewards.ndim == 1
    assert rewards.size == 8

    assert isinstance(dones, np.ndarray)
    assert dones.dtype == np.bool_
    assert dones.ndim == 1
    assert dones.size == 8


@pytest.mark.parametrize("shared_memory", [True, False])
def test_copy_async_vector_env(shared_memory):
    env_fns = [make_env("CubeCrash-v0", i) for i in range(8)]
    try:
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory, copy=True)
        observations = env.reset()
        observations[0] = 128
        assert not np.all(env.observations[0] == 128)
    finally:
        env.close()


@pytest.mark.parametrize("shared_memory", [True, False])
def test_no_copy_async_vector_env(shared_memory):
    env_fns = [make_env("CubeCrash-v0", i) for i in range(8)]
    try:
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory, copy=False)
        observations = env.reset()
        observations[0] = 128
        assert np.all(env.observations[0] == 128)
    finally:
        env.close()


@pytest.mark.parametrize("shared_memory", [True, False])
def test_reset_timeout_async_vector_env(shared_memory):
    env_fns = [make_slow_env(0.3, i) for i in range(4)]
    with pytest.raises(TimeoutError):
        try:
            env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
            env.reset_async()
            observations = env.reset_wait(timeout=0.1)
        finally:
            env.close(terminate=True)


@pytest.mark.parametrize("shared_memory", [True, False])
def test_step_timeout_async_vector_env(shared_memory):
    env_fns = [make_slow_env(0.0, i) for i in range(4)]
    with pytest.raises(TimeoutError):
        try:
            env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
            observations = env.reset()
            env.step_async([0.1, 0.1, 0.3, 0.1])
            observations, rewards, dones, _ = env.step_wait(timeout=0.1)
        finally:
            env.close(terminate=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("shared_memory", [True, False])
def test_reset_out_of_order_async_vector_env(shared_memory):
    env_fns = [make_env("CubeCrash-v0", i) for i in range(4)]
    with pytest.raises(NoAsyncCallError):
        try:
            env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
            observations = env.reset_wait()
        except NoAsyncCallError as exception:
            assert exception.name == "reset"
            raise
        finally:
            env.close(terminate=True)

    with pytest.raises(AlreadyPendingCallError):
        try:
            env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
            actions = env.action_space.sample()
            observations = env.reset()
            env.step_async(actions)
            env.reset_async()
        except NoAsyncCallError as exception:
            assert exception.name == "step"
            raise
        finally:
            env.close(terminate=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("shared_memory", [True, False])
def test_step_out_of_order_async_vector_env(shared_memory):
    env_fns = [make_env("CubeCrash-v0", i) for i in range(4)]
    with pytest.raises(NoAsyncCallError):
        try:
            env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
            actions = env.action_space.sample()
            observations = env.reset()
            observations, rewards, dones, infos = env.step_wait()
        except AlreadyPendingCallError as exception:
            assert exception.name == "step"
            raise
        finally:
            env.close(terminate=True)

    with pytest.raises(AlreadyPendingCallError):
        try:
            env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
            actions = env.action_space.sample()
            env.reset_async()
            env.step_async(actions)
        except AlreadyPendingCallError as exception:
            assert exception.name == "reset"
            raise
        finally:
            env.close(terminate=True)


@pytest.mark.parametrize("shared_memory", [True, False])
def test_already_closed_async_vector_env(shared_memory):
    env_fns = [make_env("CubeCrash-v0", i) for i in range(4)]
    with pytest.raises(ClosedEnvironmentError):
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
        env.close()
        observations = env.reset()


@pytest.mark.parametrize("shared_memory", [True, False])
def test_check_observations_async_vector_env(shared_memory):
    # CubeCrash-v0 - observation_space: Box(40, 32, 3)
    env_fns = [make_env("CubeCrash-v0", i) for i in range(8)]
    # MemorizeDigits-v0 - observation_space: Box(24, 32, 3)
    env_fns[1] = make_env("MemorizeDigits-v0", 1)
    with pytest.raises(RuntimeError):
        env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
        env.close(terminate=True)


def test_custom_space_async_vector_env():
    env_fns = [make_custom_space_env(i) for i in range(4)]
    try:
        env = AsyncVectorEnv(env_fns, shared_memory=False)
        reset_observations = env.reset()
        actions = ("action-2", "action-3", "action-5", "action-7")
        step_observations, rewards, dones, _ = env.step(actions)
    finally:
        env.close()

    assert isinstance(env.single_observation_space, CustomSpace)
    assert isinstance(env.observation_space, Tuple)

    assert isinstance(reset_observations, tuple)
    assert reset_observations == ("reset", "reset", "reset", "reset")

    assert isinstance(step_observations, tuple)
    assert step_observations == (
        "step(action-2)",
        "step(action-3)",
        "step(action-5)",
        "step(action-7)",
    )


def test_custom_space_async_vector_env_shared_memory():
    env_fns = [make_custom_space_env(i) for i in range(4)]
    with pytest.raises(ValueError):
        env = AsyncVectorEnv(env_fns, shared_memory=True)
        env.close(terminate=True)
