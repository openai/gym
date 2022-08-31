import gc
import os
import re
import time

import pytest

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class BrokenRecordableEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array_list"]}

    def __init__(self, render_mode="rgb_array_list"):
        self.render_mode = render_mode

    def render(self):
        pass


class UnrecordableEnv(gym.Env):
    metadata = {"render_modes": [None]}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

    def render(self):
        pass


def test_record_simple():
    env = gym.make(
        "CartPole-v1", render_mode="rgb_array_list", disable_env_checker=True
    )
    rec = VideoRecorder(env)
    env.reset()
    rec.capture_frame()

    rec.close()

    assert not rec.broken
    assert os.path.exists(rec.path)
    f = open(rec.path)
    assert os.fstat(f.fileno()).st_size > 100


def test_autoclose():
    def record():
        env = gym.make(
            "CartPole-v1", render_mode="rgb_array_list", disable_env_checker=True
        )
        rec = VideoRecorder(env)
        env.reset()
        rec.capture_frame()

        rec_path = rec.path

        # The function ends without an explicit `rec.close()` call
        # The Python interpreter will implicitly do `del rec` on garbage cleaning
        return rec_path

    rec_path = record()

    gc.collect()  # do explicit garbage collection for test
    time.sleep(5)  # wait for subprocess exiting

    assert os.path.exists(rec_path)
    f = open(rec_path)
    assert os.fstat(f.fileno()).st_size > 100


def test_no_frames():
    env = BrokenRecordableEnv()
    rec = VideoRecorder(env)
    rec.close()
    assert rec.functional
    assert not os.path.exists(rec.path)


def test_record_unrecordable_method():
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "\x1b[33mWARN: Disabling video recorder because environment <UnrecordableEnv instance> was not initialized with any compatible video mode between `rgb_array` and `rgb_array_list`\x1b[0m"
        ),
    ):
        env = UnrecordableEnv()
        rec = VideoRecorder(env)
        assert not rec.enabled
        rec.close()


def test_record_breaking_render_method():
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Env returned None on `render()`. Disabling further rendering for video recorder by marking as disabled:"
        ),
    ):
        env = BrokenRecordableEnv()
        rec = VideoRecorder(env)
        rec.capture_frame()
        rec.close()
        assert rec.broken
        assert not os.path.exists(rec.path)


def test_text_envs():
    env = gym.make(
        "FrozenLake-v1", render_mode="rgb_array_list", disable_env_checker=True
    )
    video = VideoRecorder(env)
    try:
        env.reset()
        video.capture_frame()
        video.close()
    finally:
        os.remove(video.path)
