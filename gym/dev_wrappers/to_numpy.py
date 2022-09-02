"""Helper functions and wrapper class for converting between numpy and Jax."""
import functools
from collections import abc
from typing import Any, Dict, Iterable, Mapping, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax.interpreters.xla import DeviceArray

from gym import Env, Wrapper


@functools.singledispatch
def numpy_to_jax(value: Any) -> Any:
    """Converts a value to a Jax DeviceArray."""
    raise Exception(
        f"No conversion for Numpy to Jax registered for type: {type(value)}"
    )


@numpy_to_jax.register(np.ndarray)
def _numpy_to_jax(value: np.ndarray) -> DeviceArray:
    """Converts a numpy array to a Jax DeviceArray."""
    return jnp.array(value)


@numpy_to_jax.register(abc.Mapping)
def _numpy_mapping_to_jax(
    value: Mapping[str, Union[np.ndarray, Any]]
) -> Mapping[str, Union[DeviceArray, Any]]:
    """Converts a dictionary of numpy arrays to a mapping of Jax DeviceArrays."""
    return type(value)(**{k: numpy_to_jax(v) for k, v in value.items()})


@numpy_to_jax.register(abc.Iterable)
def _numpy_iterable_to_jax(
    value: Iterable[Union[np.ndarray, Any]]
) -> Iterable[Union[DeviceArray, Any]]:
    """Converts an Iterable from Numpy Arrays to an iterable of Jax DeviceArrays."""
    return type(value)(numpy_to_jax(v) for v in value)


@functools.singledispatch
def jax_to_numpy(value: Any) -> Any:
    """Converts a value to a numpy array."""
    raise Exception(
        f"No conversion for Jax to Numpy registered for type: {type(value)}"
    )


@jax_to_numpy.register(DeviceArray)
def _jax_to_numpy(value: DeviceArray) -> np.ndarray:
    """Converts a Jax DeviceArray to a numpy array."""
    return np.asarray(jnp.asarray(value))


@jax_to_numpy.register(abc.Mapping)
def _jax_mapping_to_numpy(
    value: Mapping[str, Union[DeviceArray, Any]]
) -> Mapping[str, Union[np.ndarray, Any]]:
    """Converts a dictionary of Jax DeviceArrays to a mapping of numpy arrays."""
    return type(value)(**{k: jax_to_numpy(v) for k, v in value.items()})


@jax_to_numpy.register(abc.Iterable)
def _jax_iterable_to_numpy(
    value: Iterable[Union[np.ndarray, Any]]
) -> Iterable[Union[DeviceArray, Any]]:
    """Converts an Iterable from Numpy arrays to an iterable of Jax DeviceArrays."""
    return type(value)(jax_to_numpy(v) for v in value)


class JaxToNumpyV0(Wrapper):
    """Wraps an environment so that it can be interacted with through numpy arrays.

    Actions must be provided as numpy arrays and observations will be returned as numpy arrays.

    Note:
        Extensive testing has not been done for handling the array's data type.
    """

    def __init__(self, env: Union[Env, Wrapper]):
        """Wraps an environment such that the input and outputs are numpy arrays.

        Args:
            env: the environment to wrap
        """
        super().__init__(env)

    def step(
        self, action: np.ndarray
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], float, bool, Dict[str, Any]]:
        """Performs the given action in the environment.

        Args:
            action: the action to perform as a numpy array

        Returns:
            A tuple containing the next observation, reward, done, and extra info.
        """
        jax_action = numpy_to_jax(action)
        obs, reward, done, info = self.env.step(jax_action)

        obs = jax_to_numpy(obs)

        return obs, reward, done, info

    def reset(self, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """Resets the environment."""
        result = self.env.reset(**kwargs)
        if kwargs.get("return_info", False):
            return jax_to_numpy(result[0]), result[1]
        else:
            return jax_to_numpy(result)
