import contextlib
import glob
import os
import shutil
import tempfile

import gym
from gym import error
from gym.monitoring import monitor

class FakeEnv(gym.Env):
    def _render(self, close=True):
        raise RuntimeError('Raising')

@contextlib.contextmanager
def tempdir():
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)

def test_monitor_filename():
    with tempdir() as temp:
        env = gym.make('Acrobot-v0')
        env.monitor.start(temp)
        env.monitor.close()

        manifests = glob.glob(os.path.join(temp, '*.manifest.*'))
        assert len(manifests) == 1

def test_close_monitor():
    with tempdir() as temp:
        env = FakeEnv()
        env.monitor.start(temp)
        env.monitor.close()

        manifests = monitor.detect_training_manifests(temp)
        assert len(manifests) == 1

def test_video_callable():
    with tempdir() as temp:
        env = gym.make('Acrobot-v0')
        try:
            env.monitor.start(temp, video_callable=False)
        except error.Error:
            pass
        else:
            assert False

def test_env_reuse():
    with tempdir() as temp:
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
