"""
Strategies for processing the info dictionary of VecEnvs
"""
import numpy as np


class BraxInfoProcessor:
    """Process info of vectorized environment to match `Brax` format.

    Return info of a vectorized environment in the form of a single dictionary.
    Keys of the dictionary represents the `info` key; Values are lists
    in which each index correspond to an environment. If the environment
    at index `i` does not have a value for `info` then it is set to `0` for
    numeric dtpye, `None` for objects.
    To avoid ambiguity between 0's coming from actual data or 0's coming from
    no info data for the envirinoment `i`, a second boolean array with the key
    `_key` is created.

    This strategy matches Brax library info's output structure.

    Example with 3 environments in which only the last has the `terminal_observation` info:

        >>> {
        ...     "terminal_observation": np.array([
        ...         0.,
        ...         0.,
        ...         0.13)
        ...     ]),
        ...     "_terminal_observation": np.array([
        ...         False,
        ...         False,
        ...         True)
        ...     ])
        ... }
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.info = {}

    def _init_info_array(self, num_envs: int, key: str, dtype: type) -> np.ndarray:
        if dtype not in [int, float, bool]:
            dtype = object
            array = np.zeros(num_envs, dtype=dtype)
            array[:] = None
        else:
            array = np.zeros(num_envs, dtype=dtype)
        array_mask = np.zeros(self.num_envs, dtype=bool)
        return array, array_mask

    def add_info(self, info: dict, env_num: int):
        for k in info.keys():
            if k not in self.info:
                info_array, array_mask = self._init_info_array(
                    self.num_envs, k, type(info[k])
                )
            else:
                info_array, array_mask = self.info[k], self.info[f"_{k}"]

            info_array[env_num], array_mask[env_num] = info[k], True
            self.info[k], self.info[f"_{k}"] = info_array, array_mask

    def get_info(self) -> dict:
        return self.info
