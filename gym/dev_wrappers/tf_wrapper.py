from typing import Union

import jax.dlpack
import tensorflow as tf

from gym import Env, Wrapper


class TensorFlowWrapper(Wrapper):
    def __init__(self, env: Union[Env, Wrapper]):

        super().__init__(env)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def step(self, action: tf.Tensor):
        jax_action = self._tf_to_jax(action)
        obs, reward, done, info = self.env.step(jax_action)

        if self.is_vector_env:
            obs = [self._jax_to_tf(o) for o in obs]
        else:
            obs = self._jax_to_tf(obs)

        return obs, reward, done, info

    def reset(self, **kwargs) -> Union[tf.Tensor, tuple[tf.Tensor, dict]]:
        return_info = kwargs.get("return_info", False)
        if return_info:
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)

        # TODO: Verify this captures vector envs and converts everything from jax to torch
        if self.is_vector_env:
            obs = [self._jax_to_tf(o) for o in obs]
        else:
            obs = self._jax_to_tf(obs)

        if return_info:
            return obs, info
        else:
            return obs

    def _tf_to_jax(self, arr: tf.Tensor):
        return jax.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(arr))

    def _jax_to_tf(self, arr):
        return tf.experimental.dlpack.from_dlpack(jax.dlpack.to_dlpack(arr))
