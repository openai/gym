import gym
from gym import error
from gym import wrappers
from gym.wrappers import SkipWrapper

def test_skip():
    every_two_frame = SkipWrapper(2)
    env = gym.make("FrozenLake-v0")
    env = every_two_frame(env)
    obs = env.reset()
    env.render()


def test_no_double_wrapping():
    env = gym.make("FrozenLake-v0")
    env = wrappers.Monitored('/tmp', force=True)(env)

    try:
        env = wrappers.Monitored('/tmp', force=True)(env)
    except error.DoubleWrapperError:
        pass
    else:
        assert False, "Should not allow double wrapping"
