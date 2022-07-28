from typing import Any, Dict, Union, Tuple

import jax
import jax.numpy as jnp
from jax.interpreters.xla import DeviceArray

import numpy as np

from gym import Env, Wrapper
from gym.error import DependencyNotInstalled


def jax_to_numpy(arr: DeviceArray) -> np.ndarray:
    return np.asarray(jnp.asarray(arr))


def numpy_to_jax(arr: np.ndarray) -> DeviceArray:
    return jnp.array(arr)


def numpy_dict_to_jax(
    value: Dict[str, Union[np.ndarray, Any]]
) -> Dict[str, Union[DeviceArray, Any]]:
    return type(value)(**{k: numpy_to_jax(v) for k, v in value.items()})


def jax_dict_to_numpy(
    value: Dict[str, Union[DeviceArray, Any]]
) -> Dict[str, Union[np.ndarray, Any]]:
    return type(value)(**{k: jax_to_numpy(v) for k, v in value.items()})

class JaxToNumpyV0(Wrapper):
    def __init__(self, env: Union[Env, Wrapper]):
        super().__init__(env)

    def step(
        self, action: np.ndarray
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], float, bool, Dict[str, Any]]:
        jax_action = numpy_to_jax(action)
        obs, reward, done, info = self.env.step(jax_action)

        obs = jax_to_numpy(obs)

        return obs, reward, done, info

    def reset(self, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        result = self.env.reset(**kwargs)
        if kwargs.get("return_info", False):
            return jax_to_numpy(result[0]), result[1]
        else:
            return jax_to_numpy(result)
