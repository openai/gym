from typing import Any, Dict, Union, Tuple

from jax._src import dlpack as jax_dlpack
from jax.interpreters.xla import DeviceArray

from gym import Env, Wrapper
from gym.error import DependencyNotInstalled

try:
    import tensorflow as tf
except ImportError:
    raise DependencyNotInstalled(
        "tensorflow is not installed, run `pip install tensorflow`"
    )


def tf_to_jax(arr: tf.Tensor):
    # TODO: Verify this captures vector envs and converts everything from tensorflow to jax
    return jax_dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(arr))


def jax_to_tf(arr):
    # TODO: Verify this captures vector envs and converts everything from jax to tensorflow
    return tf.experimental.dlpack.from_dlpack(jax_dlpack.to_dlpack(arr))


def tf_dict_to_jax(
     value: Dict[str, Union[tf.Tensor, Any]]
) -> Dict[str, Union[DeviceArray, Any]]:
    return type(value)(**{k: tf_to_jax(v) for k, v in value.items()})


def jax_dict_to_tf(
    value: Dict[str, Union[DeviceArray, Any]]
) -> Dict[str, Union[tf.Tensor, Any]]:
    return type(value)(**{k: tf_to_jax(v) for k, v in value.items()})


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

