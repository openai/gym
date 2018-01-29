import gym
from gym import error
from gym import wrappers
import tempfile
import shutil


def test_no_double_wrapping():
    temp = tempfile.mkdtemp()
    try:
        env = gym.make("FrozenLake-v0")
        env = wrappers.Monitor(env, temp)
        try:
            env = wrappers.Monitor(env, temp)
        except error.DoubleWrapperError:
            pass
        else:
            assert False, "Should not allow double wrapping"
        env.close()
    finally:
        shutil.rmtree(temp)
