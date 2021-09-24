from functools import partial
from typing import Callable, Type
import pytest
import numpy as np
import pytest

from gym import spaces
from gym.spaces import Tuple, Box

from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.vector.vector_env import VectorEnv
from tests.vector.utils import CustomSpace, make_env


@pytest.mark.parametrize("shared_memory", [True, False])
def test_vector_env_equal(shared_memory):
    env_fns = [make_env("CartPole-v1", i) for i in range(4)]
    num_steps = 100
    try:
        async_env = AsyncVectorEnv(env_fns, shared_memory=shared_memory)
        sync_env = SyncVectorEnv(env_fns)

        assert async_env.num_envs == sync_env.num_envs
        assert async_env.observation_space == sync_env.observation_space
        assert async_env.single_observation_space == sync_env.single_observation_space
        assert async_env.action_space == sync_env.action_space
        assert async_env.single_action_space == sync_env.single_action_space

        async_observations = async_env.reset(seed=0)
        sync_observations = sync_env.reset(seed=0)
        assert np.all(async_observations == sync_observations)

        for _ in range(num_steps):
            actions = async_env.action_space.sample()
            assert actions in sync_env.action_space

            # fmt: off
            async_observations, async_rewards, async_dones, async_infos = async_env.step(actions)
            sync_observations, sync_rewards, sync_dones, sync_infos = sync_env.step(actions)
            # fmt: on

            for idx in range(len(sync_dones)):
                if sync_dones[idx]:
                    assert "terminal_observation" in async_infos[idx]
                    assert "terminal_observation" in sync_infos[idx]
                    assert sync_dones[idx]

            assert np.all(async_observations == sync_observations)
            assert np.all(async_rewards == sync_rewards)
            assert np.all(async_dones == sync_dones)

    finally:
        async_env.close()
        sync_env.close()


def test_custom_space_vector_env():
    env = VectorEnv(4, CustomSpace(), CustomSpace())

    assert isinstance(env.single_observation_space, CustomSpace)
    assert isinstance(env.observation_space, Tuple)

    assert isinstance(env.single_action_space, CustomSpace)
    assert isinstance(env.action_space, Tuple)


@pytest.mark.parametrize("base_env", ["CubeCrash-v0", "CartPole-v0"])
@pytest.mark.parametrize("async_inner", [False, True])
@pytest.mark.parametrize("async_outer", [False, True])
@pytest.mark.parametrize("n_inner_envs", [1, 4, 7])
@pytest.mark.parametrize("n_outer_envs", [1, 4, 7])
def test_nesting_vector_envs(
    base_env: str,
    async_inner: bool,
    async_outer: bool,
    n_inner_envs: int,
    n_outer_envs: int,
):
    """Tests nesting of vector envs: Using a VectorEnv of VectorEnvs.

    This can be useful for example when running a large number of environments
    on a machine with few cores, as worker process of an AsyncVectorEnv can themselves
    run multiple environments sequentially using a SyncVectorEnv (a.k.a. chunking).

    This test uses creates `n_outer_envs` vectorized environments, each of which has
    `n_inner_envs` inned environments. If `async_outer` is True, then the outermost
    wrapper is an `AsyncVectorEnv` and a `SyncVectorEnv` when `async_outer` is False.
    Same goes for the "inner" environments.

    Parameters
    ----------
    - base_env : str
        The base environment id.
    - async_inner : bool
        Wether the inner VectorEnv will be async or not.
    - async_outer : bool
        Wether the outer VectorEnv will be async or not.
    - n_inner_envs : int
        Number of inner environments.
    - n_outer_envs : int
        Number of outer environments.
    """

    inner_vectorenv_type: Type[VectorEnv] = (
        AsyncVectorEnv if async_inner else SyncVectorEnv
    )
    outer_vectorenv_type: Type[VectorEnv] = (
        partial(AsyncVectorEnv, daemon=False) if async_outer else SyncVectorEnv
    )
    # NOTE: When nesting AsyncVectorEnvs, only the "innermost" envs can have
    # `daemon=True`, otherwise the "daemonic processes are not allowed to have
    # children" AssertionError is raised in `multiprocessing.process`.

    # Create the VectorEnv of VectorEnvs
    env = outer_vectorenv_type(
        [
            partial(
                inner_vectorenv_type,
                env_fns=[
                    make_env(base_env, seed=n_inner_envs * i + j)
                    for j in range(n_inner_envs)
                ],
            )
            for i in range(n_outer_envs)
        ]
    )

    # Create a single test environment.
    with make_env(base_env, 0)() as temp_single_env:
        single_observation_space = temp_single_env.observation_space
        single_action_space = temp_single_env.action_space

    assert isinstance(single_observation_space, Box)
    assert isinstance(env.observation_space, Box)
    assert env.observation_space.shape == (
        n_outer_envs,
        n_inner_envs,
        *single_observation_space.shape,
    )
    assert env.observation_space.dtype == single_observation_space.dtype

    assert isinstance(env.action_space, spaces.Tuple)
    assert len(env.action_space.spaces) == n_outer_envs
    assert all(
        isinstance(outer_action_space, spaces.Tuple)
        and len(outer_action_space.spaces) == n_inner_envs
        for outer_action_space in env.action_space.spaces
    )
    assert all(
        [
            len(inner_action_space.spaces) == n_inner_envs
            for inner_action_space in env.action_space.spaces
        ]
    )
    assert all(
        [
            inner_action_space.spaces[i] == single_action_space
            for inner_action_space in env.action_space.spaces
            for i in range(n_inner_envs)
        ]
    )

    with env:
        observations = env.reset()
        assert observations in env.observation_space

        actions = env.action_space.sample()
        assert actions in env.action_space

        observations, rewards, dones, _ = env.step(actions)
        assert observations in env.observation_space

    assert isinstance(env.observation_space, Box)
    assert isinstance(observations, np.ndarray)
    assert observations.dtype == env.observation_space.dtype
    assert (
        observations.shape
        == (n_outer_envs, n_inner_envs) + single_observation_space.shape
    )

    assert isinstance(rewards, np.ndarray)
    assert isinstance(rewards[0], np.ndarray)
    assert rewards.ndim == 2
    assert rewards.shape == (n_outer_envs, n_inner_envs)

    assert isinstance(dones, np.ndarray)
    assert dones.dtype == np.bool_
    assert dones.ndim == 2
    assert dones.shape == (n_outer_envs, n_inner_envs)
