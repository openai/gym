import os
import shutil

import gym
from gym.utils.save_video import capped_cubic_video_schedule, save_video


def test_record_video_using_default_trigger():
    env = gym.make("CartPole-v1", render_mode="rgb_array", disable_env_checker=True)

    frames = []
    env.reset()
    for _ in range(199):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        if done:
            frames.append(env.render())
            env.reset()

    frames.append(env.render())

    save_video(frames, "videos", fps=env.metadata["render_fps"])
    env.close()
    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    assert len(mp4_files) == sum(
        capped_cubic_video_schedule(i) for i in range(len(frames))
    )
    shutil.rmtree("videos")


def test_record_video_step_trigger():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env._max_episode_steps = 20

    frames = []
    env.reset()
    for _ in range(199):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        if done:
            frames.append(env.render())
            env.reset()
    frames.append(env.render())
    env.close()

    save_video(
        frames,
        "videos",
        fps=env.metadata["render_fps"],
        step_trigger=lambda x: x % 100 == 0,
    )
    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    assert len(mp4_files) == 2
    shutil.rmtree("videos")


def make_env(gym_id, **kwargs):
    def thunk():
        env = gym.make(gym_id, disable_env_checker=True, **kwargs)
        env._max_episode_steps = 20
        return env

    return thunk


def test_record_video_within_vector():
    envs = gym.vector.SyncVectorEnv(
        [make_env("CartPole-v1", render_mode="rgb_array") for _ in range(2)]
    )
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs.reset()
    first_frames = [[]]
    for i in range(199):
        _, _, _, infos = envs.step(envs.action_space.sample())
        first_frames[-1].extend(envs.call("render")[0])

        if "episode" in infos and infos["_episode"][0]:
            first_frames.append([])

    save_video(
        first_frames,
        "videos",
        fps=envs.metadata["render_fps"],
        step_trigger=lambda x: x % 100 == 0,
    )
    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    assert len(mp4_files) == 2
    shutil.rmtree("videos")
