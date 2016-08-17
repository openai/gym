import glob
import os

import gym
from gym import error
from gym import monitoring
from gym.monitoring import monitor
from gym.monitoring.tests import helpers

class FakeEnv(gym.Env):
    def _render(self, close=True):
        raise RuntimeError('Raising')

def test_monitor_filename():
    with helpers.tempdir() as temp:
        env = gym.make('CartPole-v0')
        env.monitor.start(temp)
        env.monitor.close()

        manifests = glob.glob(os.path.join(temp, '*.manifest.*'))
        assert len(manifests) == 1

def test_write_upon_reset_false():
    with helpers.tempdir() as temp:
        env = gym.make('CartPole-v0')
        env.monitor.start(temp, video_callable=False, write_upon_reset=False)
        env.reset()

        files = glob.glob(os.path.join(temp, '*'))
        assert not files, "Files: {}".format(files)

        env.monitor.close()
        files = glob.glob(os.path.join(temp, '*'))
        assert len(files) > 0

def test_write_upon_reset_true():
    with helpers.tempdir() as temp:
        env = gym.make('CartPole-v0')
        env.monitor.start(temp, video_callable=False, write_upon_reset=True)
        env.reset()

        files = glob.glob(os.path.join(temp, '*'))
        assert len(files) > 0, "Files: {}".format(files)

        env.monitor.close()
        files = glob.glob(os.path.join(temp, '*'))
        assert len(files) > 0

def test_close_monitor():
    with helpers.tempdir() as temp:
        env = FakeEnv()
        env.monitor.start(temp)
        env.monitor.close()

        manifests = monitor.detect_training_manifests(temp)
        assert len(manifests) == 1

def test_video_callable_true_not_allowed():
    with helpers.tempdir() as temp:
        env = gym.make('CartPole-v0')
        try:
            env.monitor.start(temp, video_callable=True)
        except error.Error:
            pass
        else:
            assert False

def test_video_callable_false_does_not_record():
    with helpers.tempdir() as temp:
        env = gym.make('CartPole-v0')
        env.monitor.start(temp, video_callable=False)
        env.reset()
        env.monitor.close()
        results = monitoring.load_results(temp)
        assert len(results['videos']) == 0

def test_video_callable_records_videos():
    with helpers.tempdir() as temp:
        env = gym.make('CartPole-v0')
        env.monitor.start(temp)
        env.reset()
        env.monitor.close()
        results = monitoring.load_results(temp)
        assert len(results['videos']) == 1, "Videos: {}".format(results['videos'])

def test_env_reuse():
    with helpers.tempdir() as temp:
        env = gym.make('CartPole-v0')
        env.monitor.start(temp)
        env.monitor.close()

        env.monitor.start(temp, force=True)
        env.reset()
        env.step(env.action_space.sample())
        env.step(env.action_space.sample())
        env.monitor.close()

        results = monitor.load_results(temp)
        assert results['episode_lengths'] == [2], 'Results: {}'.format(results)
