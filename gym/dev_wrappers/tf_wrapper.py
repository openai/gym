from typing import Any, Dict, Union, Tuple

from gym import Env, Wrapper
from gym.error import DependencyNotInstalled

from gym.utils.conversions import jax_to_tf, tf_to_jax

try:
    import tensorflow as tf
except ImportError:
    raise DependencyNotInstalled(
        "tensorflow is not installed, run `pip install tensorflow`"
    )


class JaxToTFV0(Wrapper):
    def __init__(self, env: Union[Env, Wrapper]):
        """This wrapper will convert TensorFlow inputs for the actions and observations to Jax arrays
        for an underlying Jax environment then convert the return observations from Jax arrays back to
        TensorFlow tensors.

        Args:
            env: The JAx-based environment to wrap
        """
        super().__init__(env)

    def step(self, action: tf.Tensor):
        jax_action = tf_to_jax(action)
        obs, reward, done, info = self.env.step(jax_action)

        obs = jax_to_tf(obs)

        return obs, reward, done, info

    def reset(self, **kwargs) -> Union[tf.Tensor, Tuple[tf.Tensor, dict]]:
        result = self.env.reset(**kwargs)
        if kwargs.get("return_info", False):
            return jax_to_tf(result[0]), result[1]
        else:
            return jax_to_tf(result)

