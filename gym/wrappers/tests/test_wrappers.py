import gym
from gym import error
from gym import wrappers
from gym.wrappers import SkipWrapper

import tempfile
import shutil


def test_skip():
    every_two_frame = SkipWrapper(2)
    env = gym.make("FrozenLake-v0")
    env = every_two_frame(env)
    obs = env.reset()
    env.render()


def test_no_double_wrapping():
    temp = tempfile.mkdtemp()
    try:
        env = gym.make("FrozenLake-v0")
        env = wrappers.Monitor(temp)(env)
        try:
            env = wrappers.Monitor(temp)(env)
        except error.DoubleWrapperError:
            pass
        else:
            assert False, "Should not allow double wrapping"
        env.close()
    finally:
        shutil.rmtree(temp)


if __name__ == '__main__':
    test_no_double_wrapping()
