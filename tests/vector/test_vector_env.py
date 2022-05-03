import warnings
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose

import gym
from gym.envs.registration import EnvSpec, registry
from gym.spaces import Tuple
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.vector.utils.numpy_utils import concatenate
from gym.vector.utils.spaces import iterate
from gym.vector.vector_env import VectorEnv
from gym.wrappers import AutoResetWrapper
from tests.envs.spec_list import should_skip_env_spec_for_tests
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


# Only use 'local' envs for testing.
# NOTE: this won't work if the atari dependencies are installed, as we can't gym.make() them when
# inside the git repo folder.
local_env_ids = [
    spec.id for spec in registry.all() if not should_skip_env_spec_for_tests(spec)
]


def _make_seeded_env(env_id: str, seed: int) -> gym.Env:
    # Ignore any depcrecated environment warnings, since we will always need to test those.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        env = gym.make(env_id)
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env


@pytest.mark.parametrize("env_id", local_env_ids)
@pytest.mark.parametrize("async_inner", [False, True])
@pytest.mark.parametrize("async_outer", [False, True])
@pytest.mark.parametrize("n_inner_envs", [1, 4, 7])
@pytest.mark.parametrize("n_outer_envs", [1, 4, 7])
def test_nesting_vector_envs(
    env_id: str,
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
    - env_id : str
        ID of a gym environment to use as the base environment.
    - async_inner : bool
        Whether the inner VectorEnv will be async or not.
    - async_outer : bool
        Whether the outer VectorEnv will be async or not.
    - n_inner_envs : int
        Number of inner environments.
    - n_outer_envs : int
        Number of outer environments.
    """

    # NOTE: When nesting AsyncVectorEnvs, only the "innermost" envs can have
    # `daemon=True`, otherwise the "daemonic processes are not allowed to have
    # children" AssertionError is raised in `multiprocessing.process`.
    inner_vectorenv_type = AsyncVectorEnv if async_inner else SyncVectorEnv
    outer_vectorenv_type = (
        partial(AsyncVectorEnv, daemon=False) if async_outer else SyncVectorEnv
    )

    base_seed = 123

    # Create the functions for the envs at each index (i, j)
    seeds = [
        [base_seed + i * n_inner_envs + j for j in range(n_inner_envs)]
        for i in range(n_outer_envs)
    ]

    env_fns_grid = [
        [
            partial(_make_seeded_env, env_id, seed=seeds[i][j])
            for j in range(n_inner_envs)
        ]
        for i in range(n_outer_envs)
    ]

    outer_env_fns = [
        partial(
            inner_vectorenv_type,
            env_fns=env_fns_grid[i],
        )
        for i in range(n_outer_envs)
    ]

    env = outer_vectorenv_type(env_fns=outer_env_fns)

    # Note the initial obs, action, next_obs, reward, done, info in all these envs, and then
    # compare with those of the vectorenv.

    base_obs: list[list] = np.zeros([n_outer_envs, n_inner_envs]).tolist()
    base_act: list[list] = np.zeros([n_outer_envs, n_inner_envs]).tolist()
    base_next_obs: list[list] = np.zeros([n_outer_envs, n_inner_envs]).tolist()
    base_reward = np.zeros(shape=(n_outer_envs, n_inner_envs), dtype=float)
    base_done = np.zeros(shape=(n_outer_envs, n_inner_envs), dtype=bool)
    base_info: list[list[dict]] = np.zeros([n_outer_envs, n_inner_envs]).tolist()

    # Create an env temporarily to get the observation and action spaces.
    with env_fns_grid[0][0]() as temp_env:
        base_observation_space = temp_env.observation_space
        base_action_space = temp_env.action_space

    # Go through each index (i, j) and create the env with the seed at that index, getting the
    # initial state, action, next_obs, reward, done, info, etc.
    # This will then be compared with the states produced by the VectorEnv equivalent.

    for i in range(n_outer_envs):
        for j in range(n_inner_envs):
            # Create a properly seeded environment. Then, reset, and step once.
            with env_fns_grid[i][j]() as temp_env:

                # Add the AutoResetWrapper to the individual environments to replicate what will
                # happen in the VectorEnv. (See the note below).
                temp_env = AutoResetWrapper(temp_env)

                assert temp_env.observation_space == base_observation_space
                assert temp_env.action_space == base_action_space

                # NOTE: This will change a bit once the AutoResetWrapper is used in the VectorEnvs.
                base_obs[i][j], base_info[i][j] = temp_env.reset(
                    seed=seeds[i][j], return_info=True
                )
                base_act[i][j] = base_action_space.sample()
                (
                    base_next_obs[i][j],
                    base_reward[i][j],
                    base_done[i][j],
                    base_info[i][j],
                ) = temp_env.step(base_act[i][j])

    obs = env.reset(seed=seeds)

    # NOTE: creating these values so they aren't possibly unbound below and type hinters can relax.
    i = -1
    j = -1

    for i, obs_i in enumerate(iterate(env.observation_space, obs)):
        for j, obs_ij in enumerate(iterate(env.single_observation_space, obs_i)):
            assert obs_ij in base_observation_space
            # Assert that each observation is what we'd expect (following the single env.)
            assert_allclose(obs_ij, base_obs[i][j])

        assert j == n_inner_envs - 1
    assert i == n_outer_envs - 1

    # NOTE: Sampling an action using env.action_space.sample() would give a different value than
    # if we sampled actions from each env individually and batched them.
    # In order to check that everything is working correctly, we'll instead create the action by
    # concatenating the individual actions, and pass it to the vectorenv, to check if that will
    # recreate the same result for all individual envs.
    # _ = env.action_space.sample()
    action = concatenate(
        env.single_action_space,
        [
            concatenate(base_action_space, base_act[i], out=None)
            for i in range(n_outer_envs)
        ],
        out=None,
    )

    for i, action_i in enumerate(iterate(env.action_space, action)):
        for j, action_ij in enumerate(iterate(env.single_action_space, action_i)):
            assert action_ij in base_action_space
            # Assert that each observation is what we'd expect (following the single env.)
            # assert_allclose(act_ij, base_act)
        assert j == n_inner_envs - 1
    assert i == n_outer_envs - 1

    # Perform a single step:

    next_obs, reward, done, info = env.step(action)

    for i, next_obs_i in enumerate(iterate(env.observation_space, next_obs)):
        for j, next_obs_ij in enumerate(
            iterate(env.single_observation_space, next_obs_i)
        ):
            assert next_obs_ij in base_observation_space
            # Assert that each next observation is what we'd expect (following the single env.)
            assert_allclose(next_obs_ij, base_next_obs[i][j])

    for i, rew_i in enumerate(reward):
        for j, rew_ij in enumerate(rew_i):
            # Assert that each reward is what we'd expect (following the single env.)
            assert_allclose(rew_ij, base_reward[i][j])
        assert j == n_inner_envs - 1
    assert i == n_outer_envs - 1

    for i, done_i in enumerate(done):
        for j, done_ij in enumerate(done_i):
            assert done_ij == base_done[i][j]
        assert j == n_inner_envs - 1
    assert i == n_outer_envs - 1

    for i, info_i in enumerate(info):
        for j, info_ij in enumerate(info_i):
            # NOTE: Since the VectorEnvs don't apply an AutoResetWrapper to the individual envs,
            # the autoreset logic is in the 'worker' code, and this code doesn't add the
            # 'terminal_info' entry in the 'info' dictionary.
            # NOTE: This test-case is forward-compatible in case the VectorEnvs do end up adding
            # the 'terminal_info' entry in the 'info' dictionary.
            expected_info = base_info[i][j].copy()
            if (
                info_ij != base_info[i][j]
                and ("terminal_info" in expected_info)
                and ("terminal_info" not in info_ij)
            ):
                # Remove the 'terminal_info' key from the expected info dict and compare as before.
                expected_info.pop("terminal_info")
            assert info_ij == expected_info
        assert j == n_inner_envs - 1
    assert i == n_outer_envs - 1

    env.close()
