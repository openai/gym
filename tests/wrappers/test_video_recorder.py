import gc
import os
import time

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder


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
    proc = rec.encoder.proc

    assert proc.poll() is None  # subprocess is running

    rec.close()

    assert proc.poll() is not None  # subprocess is terminated
    assert not rec.empty
    assert not rec.broken
    assert os.path.exists(rec.path)
    f = open(rec.path)
    assert os.fstat(f.fileno()).st_size > 100


def test_autoclose():
    def record():
        env = gym.make("CartPole-v1")
        rec = VideoRecorder(env)
        env.reset()
        rec.capture_frame()

        rec_path = rec.path
        proc = rec.encoder.proc

        assert proc.poll() is None  # subprocess is running

        # The function ends without an explicit `rec.close()` call
        # The Python interpreter will implicitly do `del rec` on garbage cleaning
        return rec_path, proc

    rec_path, proc = record()

    gc.collect()  # do explicit garbage collection for test
    time.sleep(5)  # wait for subprocess exiting

    assert proc.poll() is not None  # subprocess is terminated
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
