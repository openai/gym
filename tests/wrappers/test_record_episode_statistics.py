import pytest

import gym
from gym.wrappers import RecordEpisodeStatistics


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
@pytest.mark.parametrize("deque_size", [2, 5])
def test_record_episode_statistics(env_id, deque_size):
    env = gym.make(env_id, return_two_dones=True)
    env = RecordEpisodeStatistics(env, deque_size)

    for n in range(5):
        env.reset()
        assert env.episode_returns[0] == 0.0
        assert env.episode_lengths[0] == 0
        for t in range(env.spec.max_episode_steps):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                assert "episode" in info
                assert all([item in info["episode"] for item in ["r", "l", "t"]])
                break
    assert len(env.return_queue) == deque_size
    assert len(env.length_queue) == deque_size


def test_record_episode_statistics_reset_info():
    env = gym.make("CartPole-v1", return_two_dones=True)
    env = RecordEpisodeStatistics(env)
    ob_space = env.observation_space
    obs = env.reset()
    assert ob_space.contains(obs)
    del obs
    obs, info = env.reset(return_info=True)
    assert ob_space.contains(obs)
    assert isinstance(info, dict)


@pytest.mark.parametrize(
    ("num_envs", "asynchronous"), [(1, False), (1, True), (4, False), (4, True)]
)
def test_record_episode_statistics_with_vectorenv(num_envs, asynchronous):
    envs = gym.vector.make(
        "CartPole-v1",
        num_envs=num_envs,
        asynchronous=asynchronous,
        return_two_dones=True,
    )
    envs = RecordEpisodeStatistics(envs)
    max_episode_step = (
        envs.env_fns[0]().spec.max_episode_steps
        if asynchronous
        else envs.env.envs[0].spec.max_episode_steps
    )
    envs.reset()
    for _ in range(max_episode_step + 1):
        _, _, terminateds, truncateds, infos = envs.step(envs.action_space.sample())
        for idx, info in enumerate(infos):
            if terminateds[idx] or truncateds[idx]:
                assert "episode" in info
                assert all([item in info["episode"] for item in ["r", "l", "t"]])
                break
