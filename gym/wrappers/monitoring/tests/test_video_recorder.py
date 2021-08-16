import json
import os
import shutil
import tempfile
import numpy as np

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder, video_recorder_closer


class BrokenRecordableEnv(object):
    metadata = {"render.modes": [None, "rgb_array"]}

    def render(self, mode=None):
        pass


class UnrecordableEnv(object):
    metadata = {"render.modes": [None]}

    def render(self, mode=None):
        pass


def test_record_simple():
    env = gym.make("CartPole-v1")
    rec = VideoRecorder(env)
    env.reset()
    rec.capture_frame()
    rec.close()
    assert not rec.empty
    assert not rec.broken
    assert os.path.exists(rec.path)
    f = open(rec.path)
    assert os.fstat(f.fileno()).st_size > 100


def test_autoclose():
    env = gym.make("CartPole-v1")
    rec = VideoRecorder(env)
    env.reset()
    rec.capture_frame()

    rec_path = rec.path
    with video_recorder_closer.lock:
        num_registered = len(video_recorder_closer.closeables)
    del rec

    with video_recorder_closer.lock:
        assert len(video_recorder_closer.closeables) == num_registered - 1
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


def test_record_breaking_render_method():
    env = BrokenRecordableEnv()
    rec = VideoRecorder(env)
    rec.capture_frame()
    rec.close()
    assert rec.empty
    assert rec.broken
    assert not os.path.exists(rec.path)


def test_text_envs():
    env = gym.make("FrozenLake-v1")
    video = VideoRecorder(env)
    try:
        env.reset()
        video.capture_frame()
        video.close()
    finally:
        os.remove(video.path)
