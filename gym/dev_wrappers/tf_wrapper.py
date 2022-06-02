from typing import Union

import jax.dlpack

from gym import Env, Wrapper
from gym.error import DependencyNotInstalled

try:
    import tensorflow as tf
except ImportError:
    raise DependencyNotInstalled(
        "tensorflow is not installed, run `pip install tensorflow`"
    )


class jax_to_tf_v0(Wrapper):
    def __init__(self, env: Union[Env, Wrapper]):
        """This wrapper will convert TensorFlow inputs for the actions and observations to Jax arrays
        for an underlying Jax environment then convert the return observations from Jax arrays back to
        TensorFlow tensors.

        Args:
            env: The JAx-based environment to wrap
        """
        super().__init__(env)

    def step(self, action: tf.Tensor):
        jax_action = self._tf_to_jax(action)
        obs, reward, done, info = self.env.step(jax_action)

        obs = self._jax_to_tf(obs)

        return obs, reward, done, info

    def reset(self, **kwargs) -> Union[tf.Tensor, tuple[tf.Tensor, dict]]:
        result = self.env.reset(**kwargs)
        if kwargs.get("return_info", False):
            return self._jax_to_tf(result[0]), result[1]
        else:
            return self._jax_to_tf(result)

    def _tf_to_jax(self, arr: tf.Tensor):
        # TODO: Verify this captures vector envs and converts everything from tensorflow to jax
        return jax.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(arr))

    def _jax_to_tf(self, arr):
        # TODO: Verify this captures vector envs and converts everything from jax to tensorflow
        return tf.experimental.dlpack.from_dlpack(jax.dlpack.to_dlpack(arr))
