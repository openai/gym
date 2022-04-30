"""
Strategies for processing the info dictionary of VecEnvs
"""
import warnings
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from gym.error import InvalidInfoFormat, InvalidInfoStrategy


class VecEnvInfoStrategy(ABC):
    """Interface for implementing a info
    processing strategy for vectorized environments
    """

    @abstractmethod
    def __init__(self, num_envs: int):
        ...

    @abstractmethod
    def add_info(self, info: dict, env_num: int):
        """Get the info dict from the
        environment and process with the defined strategy
        """

    @abstractmethod
    def get_info(self):
        """Return the info for the
        vectorized env
        """


class ClassicVecEnvInfoStrategy(VecEnvInfoStrategy):
    """Process the info dictionary of an environment
    so that the vectorized info is returned in the form of a list of dictionaries.

    Example with 3 environments:
        [{}, {}, {}]
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.info = []

    def add_info(self, info: dict, env_num: int):
        self.info.append(info)

    def get_info(self) -> list:
        return self.info


class BraxVecEnvInfoStrategy(VecEnvInfoStrategy):
    """Process the info dictionary of an environment
    so that the vectorized info is returned in the form of a single dictionary.
    Keys of the dictionary represents the `info` key; Values are lists
    in which each index correspond to an environment. If the environment
    at index `i` does not have a value for `info` then it is set to `None`

    This strategy matches Brax library info's output structure.

    Example with 3 environments in which only the last has the `terminal_observation` info:

        >>> {
        ...     "terminal_observation": np.array([
        ...         None,
        ...         None,
        ...         array([0.13,  1.58 , -0.22, -2.56])
        ...     ])
        ... }
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.info = {}

    def _init_info_array(self, num_envs: int, key: str, dtype: type) -> np.ndarray:
        if dtype == bool:
            warnings.warn(
                f"found boolean dtype in info for key `{key}`, empty values are set to `False`"
            )
        if dtype not in [int, float, bool]:
            dtype = object
            array = np.zeros(num_envs, dtype=dtype)
            array[:] = None
        else:
            array = np.zeros(num_envs, dtype=dtype)
        return array

    def add_info(self, info: dict, env_num: int):
        for k in info.keys():
            info_array = self.info.get(
                k, self._init_info_array(self.num_envs, k, type(info[k]))
            )
            info_array[env_num] = info[k]
            self.info[k] = info_array

    def get_info(self) -> dict:
        return self.info


class InfoStrategiesEnum(Enum):
    classic: str = "classic"
    brax: str = "brax"


class InfoStrategyFactory:
    _strategies = {
        InfoStrategiesEnum.classic.value: ClassicVecEnvInfoStrategy,
        InfoStrategiesEnum.brax.value: BraxVecEnvInfoStrategy,
    }

    def get_info_strategy(info_format: str) -> VecEnvInfoStrategy:
        if info_format not in InfoStrategyFactory._strategies:
            raise InvalidInfoFormat(
                f"{info_format} is not an available format for info, please choose one between {list(InfoStrategyFactory._strategies.keys())}"
            )
        return InfoStrategyFactory._strategies[info_format]

    def add_info_strategy(name: str, strategy: VecEnvInfoStrategy) -> None:
        if not issubclass(strategy, VecEnvInfoStrategy):
            raise InvalidInfoStrategy(
                "The strategy need to subclass the Abstract Base Class `VecEnvInfoStrategy`"
            )
        InfoStrategyFactory._strategies[name] = strategy
