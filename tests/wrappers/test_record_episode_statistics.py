import pytest

import gym
from gym.wrappers import RecordEpisodeStatistics


@pytest.mark.parametrize("env_id", ["CartPole-v0", "Pendulum-v1"])
@pytest.mark.parametrize("deque_size", [2, 5])
def test_record_episode_statistics(env_id, deque_size):
    env = gym.make(env_id)
    env = RecordEpisodeStatistics(env, deque_size)

    for n in range(5):
        env.reset()
        assert env.episode_returns[0] == 0.0
        assert env.episode_lengths[0] == 0
        for t in range(env.spec.max_episode_steps):
            _, _, done, info = env.step(env.action_space.sample())
            if done:
                assert "episode" in info
                assert all([item in info["episode"] for item in ["r", "l", "t"]])
                break
    assert len(env.return_queue) == deque_size
    assert len(env.length_queue) == deque_size


@pytest.mark.parametrize("num_envs", [1, 4])
def test_record_episode_statistics_with_vectorenv(num_envs):
    envs = gym.vector.make("CartPole-v0", num_envs=num_envs, asynchronous=False)
    envs = RecordEpisodeStatistics(envs)
    envs.reset()
    for _ in range(envs.env.envs[0].spec.max_episode_steps + 1):
        _, _, dones, infos = envs.step(envs.action_space.sample())
        for idx, info in enumerate(infos):
            if dones[idx]:
                assert "episode" in info
                assert all([item in info["episode"] for item in ["r", "l", "t"]])
                break
