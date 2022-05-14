import pytest

import gym
from gym.wrappers import ClassicVectorInfo, RecordEpisodeStatistics

ENV_ID = "CartPole-v1"
NUM_ENVS = 3
ENV_STEPS = 50


def test_usage_in_vector_env():
    env = gym.make(ENV_ID)
    vector_env = gym.vector.make(ENV_ID, num_envs=NUM_ENVS)

    ClassicVectorInfo(vector_env)

    with pytest.raises(AssertionError):
        ClassicVectorInfo(env)


def test_info_to_classic():
    env_to_wrap = gym.vector.make(ENV_ID, num_envs=NUM_ENVS)
    wrapped_env = ClassicVectorInfo(env_to_wrap)
    _, info = wrapped_env.reset(return_info=True)
    assert isinstance(info, list)
    assert len(info) == NUM_ENVS

    for _ in range(ENV_STEPS):
        action = wrapped_env.action_space.sample()
        _, _, dones, info_classic = wrapped_env.step(action)
        for i, done in enumerate(dones):
            if done:
                assert "terminal_observation" in info_classic[i]
            else:
                assert "terminal_observation" not in info_classic[i]


def test_info_to_classic_statistics():
    env_to_wrap = gym.vector.make(ENV_ID, num_envs=NUM_ENVS)
    wrapped_env = ClassicVectorInfo(RecordEpisodeStatistics(env_to_wrap))
    _, info = wrapped_env.reset(return_info=True)
    assert isinstance(info, list)
    assert len(info) == NUM_ENVS

    for _ in range(ENV_STEPS):
        action = wrapped_env.action_space.sample()
        _, _, dones, info_classic = wrapped_env.step(action)
        for i, done in enumerate(dones):
            if done:
                assert "episode" in info_classic[i]
                for stats in ["r", "l", "t"]:
                    assert stats in info_classic[i]["episode"]
                    assert isinstance(info_classic[i]["episode"][stats], float)
            else:
                assert "episode" not in info_classic[i]
