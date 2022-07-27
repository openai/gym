import numpy as np
import pytest

from gym.spaces import Tuple
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.vector.vector_env import VectorEnv
from tests.vector.utils import CustomSpace, make_env


@pytest.mark.parametrize("shared_memory", [True, False])
def test_vector_env_equal(shared_memory):
    env_fns = [make_env("CartPole-v1", i) for i in range(4)]
    num_steps = 100

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

        if any(sync_dones):
            assert "final_observation" in async_infos
            assert "_final_observation" in async_infos
            assert "final_observation" in sync_infos
            assert "_final_observation" in sync_infos

        assert np.all(async_observations == sync_observations)
        assert np.all(async_rewards == sync_rewards)
        assert np.all(async_dones == sync_dones)

    async_env.close()
    sync_env.close()


def test_custom_space_vector_env():
    env = VectorEnv(4, CustomSpace(), CustomSpace())

    assert isinstance(env.single_observation_space, CustomSpace)
    assert isinstance(env.observation_space, Tuple)

    assert isinstance(env.single_action_space, CustomSpace)
    assert isinstance(env.action_space, Tuple)
