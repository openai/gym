import os
import shutil

import numpy as np

import gym
from gym.utils.save_video import capped_cubic_video_schedule, save_video


def test_record_video_using_default_trigger():
    env = gym.make(
        "CartPole-v1", render_mode="rgb_array_list", disable_env_checker=True
    )

    env.reset()
    step_starting_index = 0
    episode_index = 0
    for step_index in range(199):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            save_video(
                env.render(),
                "videos",
                fps=env.metadata["render_fps"],
                step_starting_index=step_starting_index,
                episode_index=episode_index,
            )
            step_starting_index = step_index + 1
            episode_index += 1
            env.reset()

    env.close()
    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    shutil.rmtree("videos")
    assert len(mp4_files) == sum(
        capped_cubic_video_schedule(i) for i in range(episode_index)
    )


def modulo_step_trigger(mod: int):
    def step_trigger(step_index):
        return step_index % mod == 0

    return step_trigger


def test_record_video_step_trigger():
    env = gym.make("CartPole-v1", render_mode="rgb_array_list")
    env._max_episode_steps = 20

    env.reset()
    step_starting_index = 0
    episode_index = 0
    for step_index in range(199):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            save_video(
                env.render(),
                "videos",
                fps=env.metadata["render_fps"],
                step_trigger=modulo_step_trigger(100),
                step_starting_index=step_starting_index,
                episode_index=episode_index,
            )
            step_starting_index = step_index + 1
            episode_index += 1
            env.reset()
    env.close()

    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    shutil.rmtree("videos")
    assert len(mp4_files) == 2


def test_record_video_within_vector():
    step_trigger = modulo_step_trigger(100)
    n_steps = 199
    expected_video = 2

    envs = gym.vector.make(
        "CartPole-v1", num_envs=2, asynchronous=True, render_mode="rgb_array_list"
    )
    envs.reset()
    episode_frames = []
    step_starting_index = 0
    episode_index = 0
    for step_index in range(n_steps):
        _, _, terminated, truncated, _ = envs.step(envs.action_space.sample())
        episode_frames.extend(envs.call("render")[0])

        if np.any(np.logical_or(terminated, truncated)):
            save_video(
                episode_frames,
                "videos",
                fps=envs.metadata["render_fps"],
                step_trigger=step_trigger,
                step_starting_index=step_starting_index,
                episode_index=episode_index,
            )
            episode_frames = []
            step_starting_index = step_index + 1
            episode_index += 1

            # TODO: fix this test (see https://github.com/openai/gym/issues/3054)
            if step_trigger(step_index):
                expected_video -= 1

    envs.close()

    assert os.path.isdir("videos")
    mp4_files = [file for file in os.listdir("videos") if file.endswith(".mp4")]
    shutil.rmtree("videos")
    assert len(mp4_files) == expected_video
