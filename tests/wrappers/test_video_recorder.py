import gc
import os
import time

import pytest

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class BrokenRecordableEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode="rgb_array"):
        self.render_mode = render_mode

    def render(self, mode="human"):
        pass


class UnrecordableEnv(gym.Env):
    metadata = {"render_modes": [None]}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

    def render(self, mode="human"):
        pass


def test_record_simple():
    env = gym.make("CartPole-v1", render_mode="rgb_array", disable_env_checker=True)
    rec = VideoRecorder(env)
    env.reset()
    rec.capture_frame()
    assert rec.encoder is not None
    proc = rec.encoder.proc

    assert proc is not None and proc.poll() is None  # subprocess is running

    rec.close()

    assert proc.poll() is not None  # subprocess is terminated
    assert not rec.empty
    assert not rec.broken
    assert os.path.exists(rec.path)
    f = open(rec.path)
    assert os.fstat(f.fileno()).st_size > 100


def test_autoclose():
    def record():
        env = gym.make("CartPole-v1", render_mode="rgb_array", disable_env_checker=True)
        rec = VideoRecorder(env)
        env.reset()
        rec.capture_frame()

        rec_path = rec.path
        assert rec.encoder is not None
        proc = rec.encoder.proc

        assert proc is not None and proc.poll() is None  # subprocess is running

        # The function ends without an explicit `rec.close()` call
        # The Python interpreter will implicitly do `del rec` on garbage cleaning
        return rec_path, proc

    rec_path, proc = record()

    gc.collect()  # do explicit garbage collection for test
    time.sleep(5)  # wait for subprocess exiting

    assert proc is not None and proc.poll() is not None  # subprocess is terminated
    assert os.path.exists(rec_path)
    f = open(rec_path)
    assert os.fstat(f.fileno()).st_size > 100


def test_no_frames():
    env = BrokenRecordableEnv()
    rec = VideoRecorder(env)
    rec.close()
    assert rec.empty
    assert rec.functional
    assert not os.path.exists(rec.path)


def test_record_unrecordable_method():
    env = UnrecordableEnv()
    rec = VideoRecorder(env)
    assert not rec.enabled
    rec.close()


@pytest.mark.filterwarnings("ignore:.*Env returned None on render.*")
def test_record_breaking_render_method():
    env = BrokenRecordableEnv()
    rec = VideoRecorder(env)
    rec.capture_frame()
    rec.close()
    assert rec.empty
    assert rec.broken
    assert not os.path.exists(rec.path)


def test_text_envs():
    env = gym.make("FrozenLake-v1", render_mode="rgb_array", disable_env_checker=True)
    video = VideoRecorder(env)
    try:
        env.reset()
        video.capture_frame()
        video.close()
    finally:
        os.remove(video.path)
