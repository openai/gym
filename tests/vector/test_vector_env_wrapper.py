import numpy as np

from gym.vector import VectorEnvWrapper, make


class DummyWrapper(VectorEnvWrapper):
    def __init__(self, env):
        self.env = env
        self.counter = 0

    def reset_async(self, **kwargs):
        super().reset_async()
        self.counter += 1


def test_vector_env_wrapper_inheritance():
    env = make("FrozenLake-v1", asynchronous=False)
    wrapped = DummyWrapper(env)
    wrapped.reset()
    assert wrapped.counter == 1


def test_vector_env_wrapper_attributes():
    """Test if `set_attr`, `call` methods for VecEnvWrapper get correctly forwarded to the vector env it is wrapping."""
    env = make("CartPole-v1", num_envs=3)
    wrapped = DummyWrapper(make("CartPole-v1", num_envs=3))

    assert np.allclose(wrapped.call("gravity"), env.call("gravity"))
    env.set_attr("gravity", [20.0, 20.0, 20.0])
    wrapped.set_attr("gravity", [20.0, 20.0, 20.0])
    assert np.allclose(wrapped.get_attr("gravity"), env.get_attr("gravity"))
