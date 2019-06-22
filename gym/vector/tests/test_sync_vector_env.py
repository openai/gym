import pytest
import numpy as np

from gym.spaces import Box
from gym.vector.tests.utils import make_env, make_slow_env

from gym.vector.sync_vector_env import SyncVectorEnv

def test_create_sync_vector_env():
    env_fns = [make_env('CubeCrash-v0', i) for i in range(8)]
    try:
        env = SyncVectorEnv(env_fns)
    finally:
        env.close()

    assert env.num_envs == 8


def test_reset_sync_vector_env():
    env_fns = [make_env('CubeCrash-v0', i) for i in range(8)]
    try:
        env = SyncVectorEnv(env_fns)
        observations = env.reset()
    finally:
        env.close()

    assert isinstance(env.observation_space, Box)
    assert isinstance(observations, np.ndarray)
    assert observations.dtype == env.observation_space.dtype
    assert observations.shape == (8,) + env.single_observation_space.shape
    assert observations.shape == env.observation_space.shape


@pytest.mark.parametrize('use_single_action_space', [True, False])
def test_step_sync_vector_env(use_single_action_space):
    env_fns = [make_env('CubeCrash-v0', i) for i in range(8)]
    try:
        env = SyncVectorEnv(env_fns)
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


def test_check_observations_sync_vector_env():
    # CubeCrash-v0 - observation_space: Box(40, 32, 3)
    env_fns = [make_env('CubeCrash-v0', i) for i in range(8)]
    # MemorizeDigits-v0 - observation_space: Box(24, 32, 3)
    env_fns[1] = make_env('MemorizeDigits-v0', 1)
    with pytest.raises(RuntimeError):
        env = SyncVectorEnv(env_fns)
        env.close()


def test_episodic_sync_vector_env():
    episode_lengths = [2, 5, 3, 1, 3]
    env_fns = [make_slow_env(0., i, length) for i, length
               in enumerate(episode_lengths)]
    try:
        env = SyncVectorEnv(env_fns, episodic=True)
        observations = env.reset()
        # Step 1
        actions = env.action_space.sample()
        observations, rewards, dones, infos = env.step(actions)
        assert dones[3]
        assert np.all(observations[3] == 0)
        assert 'SyncVectorEnv.end_episode' in infos[3]
        assert infos[3]['SyncVectorEnv.end_episode']
        assert not np.all(dones)
        assert np.any(observations[0] != 0)

        # Step 2
        actions = env.action_space.sample()
        observations, rewards, dones, infos = env.step(actions)
        for j in [0, 3]:
            assert dones[j]
            assert np.all(observations[j] == 0)
        assert 'SyncVectorEnv.end_episode' in infos[0]
        assert infos[0]['SyncVectorEnv.end_episode']
        assert not np.all(dones)
        assert np.any(observations[2] != 0)

        # Step 3
        actions = env.action_space.sample()
        observations, rewards, dones, infos = env.step(actions)
        for j in [0, 2, 3, 4]:
            assert dones[j]
            assert np.all(observations[j] == 0)
        assert 'SyncVectorEnv.end_episode' in infos[2]
        assert infos[2]['SyncVectorEnv.end_episode']
        assert not np.all(dones)
        assert np.any(observations[1] != 0)

        # Step 4
        actions = env.action_space.sample()
        observations, rewards, dones, infos = env.step(actions)
        for j in [0, 2, 3, 4]:
            assert dones[j]
            assert np.all(observations[j] == 0)
        assert 'SyncVectorEnv.end_episode' not in infos[2]
        assert not np.all(dones)
        assert np.any(observations[1] != 0)

        # Step 5
        actions = env.action_space.sample()
        observations, rewards, dones, infos = env.step(actions)
        assert np.all(dones)
        assert np.all(observations == 0)
    finally:
        env.close()
