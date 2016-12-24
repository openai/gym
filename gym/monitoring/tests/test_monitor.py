import glob
import os

import gym
from gym import error, spaces
from gym import monitoring
from gym.monitoring.tests import helpers
from gym.wrappers import Monitor


def test_monitor_filename():
    with helpers.tempdir() as temp:
        env = gym.make('CartPole-v0')
        env = Monitor(directory=temp)(env)
        env.close()

        manifests = glob.glob(os.path.join(temp, '*.manifest.*'))
        assert len(manifests) == 1

def test_write_upon_reset_false():
    with helpers.tempdir() as temp:
        env = gym.make('CartPole-v0')
        env = Monitor(directory=temp, video_callable=False, write_upon_reset=False)(env)
        env.reset()

        files = glob.glob(os.path.join(temp, '*'))
        assert not files, "Files: {}".format(files)

        env.close()
        files = glob.glob(os.path.join(temp, '*'))
        assert len(files) > 0

def test_write_upon_reset_true():
    with helpers.tempdir() as temp:
        env = gym.make('CartPole-v0')
        env = Monitor(directory=temp, video_callable=False, write_upon_reset=True)(env)
        env.reset()

        files = glob.glob(os.path.join(temp, '*'))
        assert len(files) > 0, "Files: {}".format(files)

        env.close()
        files = glob.glob(os.path.join(temp, '*'))
        assert len(files) > 0

def test_video_callable_true_not_allowed():
    with helpers.tempdir() as temp:
        env = gym.make('CartPole-v0')
        try:
            env = Monitor(temp, video_callable=True)(env)
        except error.Error:
            pass
        else:
            assert False

def test_video_callable_false_does_not_record():
    with helpers.tempdir() as temp:
        env = gym.make('CartPole-v0')
        env = Monitor(temp, video_callable=False)(env)
        env.reset()
        env.close()
        results = monitoring.load_results(temp)
        assert len(results['videos']) == 0

def test_video_callable_records_videos():
    with helpers.tempdir() as temp:
        env = gym.make('CartPole-v0')
        env = Monitor(temp)(env)
        env.reset()
        env.close()
        results = monitoring.load_results(temp)
        assert len(results['videos']) == 1, "Videos: {}".format(results['videos'])

def test_semisuper_succeeds():
    """Regression test. Ensure that this can write"""
    with helpers.tempdir() as temp:
        env = gym.make('SemisuperPendulumDecay-v0')
        env = Monitor(temp)(env)
        env.reset()
        env.step(env.action_space.sample())
        env.close()

class AutoresetEnv(gym.Env):
    metadata = {'semantics.autoreset': True}

    def __init__(self):
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(1)

    def _reset(self):
        return 0

    def _step(self, action):
        return 0, 0, False, {}

import logging
logger = logging.getLogger()
gym.envs.register(
    id='Autoreset-v0',
    entry_point='gym.monitoring.tests.test_monitor:AutoresetEnv',
    timestep_limit=2,
)
def test_env_reuse():
    with helpers.tempdir() as temp:
        env = gym.make('Autoreset-v0')
        env = Monitor(temp)(env)

        env.reset()

        env.step(None)
        _, _, done, _ = env.step(None)
        assert done

        env.step(None)
        _, _, done, _ = env.step(None)
        assert done

        env.close()

def test_no_monitor_reset_unless_done():
    def assert_reset_raises(env):
        errored = False
        try:
            env.reset()
        except error.Error:
            errored = True
        assert errored, "Env allowed a reset when it shouldn't have"

    with helpers.tempdir() as temp:
        # Make sure we can reset as we please without monitor
        env = gym.make('CartPole-v0')
        env.reset()
        env.step(env.action_space.sample())
        env.step(env.action_space.sample())
        env.reset()

        # can reset once as soon as we start
        env = Monitor(temp, video_callable=False)(env)
        env.reset()

        # can reset multiple times in a row
        env.reset()
        env.reset()

        env.step(env.action_space.sample())
        env.step(env.action_space.sample())
        assert_reset_raises(env)

        # should allow resets after the episode is done
        d = False
        while not d:
            _, _, d, _ = env.step(env.action_space.sample())

        env.reset()
        env.reset()

        env.step(env.action_space.sample())
        assert_reset_raises(env)

        env.close()

def test_only_complete_episodes_written():
    with helpers.tempdir() as temp:
        env = gym.make('CartPole-v0')
        env = Monitor(temp, video_callable=False)(env)
        env.reset()
        d = False
        while not d:
            _, _, d, _ = env.step(env.action_space.sample())

        env.reset()
        env.step(env.action_space.sample())

        env.close()

        # Only 1 episode should be written
        results = monitoring.load_results(temp)
        assert len(results['episode_lengths']) == 1, "Found {} episodes written; expecting 1".format(len(results['episode_lengths']))
