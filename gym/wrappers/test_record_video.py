import pytest

import os
import shutil
import gym
from gym.wrappers import RecordEpisodeStatistics, RecordVideo


def test_record_video():
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordVideo(
        env, "videos", record_video_trigger=lambda x: x % 100 == 0
    )
    env.reset()
    for _ in range(199):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        if done:
            env.reset()
    env.close()
    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    assert len(mp4_files) == 2
    shutil.rmtree("videos")


def make_env(gym_id, seed):
    def thunk():
        env = gym.make(gym_id)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        if seed == 1:
            env = gym.wrappers.RecordVideo(
                env, "videos", record_video_trigger=lambda x: x % 100 == 0
            )
        return env

    return thunk


def test_record_video_within_vector():
    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", 1 + i) for i in range(2)])
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs.reset()
    for i in range(199):
        _, _, _, infos = envs.step(envs.action_space.sample())
        for info in infos:
            if "episode" in info.keys():
                print(f"i, episode_reward={info['episode']['r']}")
                break
    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    assert len(mp4_files) == 2
    shutil.rmtree("videos")
