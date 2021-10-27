import gym
from gym.vector import make
from gym.vector import VectorEnvWrapper


class DummyWrapper(VectorEnvWrapper):
    def __init__(self, env):
        self.env = env
        self.counter = 0

    def reset_async(self):
        super().reset_async()
        self.counter += 1


def test_vector_env_wrapper_inheritance():
    env = make("FrozenLake-v1", asynchronous=False)
    wrapped = DummyWrapper(env)
    wrapped.reset()
    assert wrapped.counter == 1
