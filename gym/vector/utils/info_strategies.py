"""
Strategies for processing the info dictionary of VecEnvs
"""
from enum import Enum


class ClassicVecEnvInfoStrategy:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.info = []

    def add_info(self, info: dict, env_num: int):
        self.info.append(info)

    def get_info(self) -> list:
        return self.info


class BraxVecEnvInfoStrategy:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.info = {}

    def add_info(self, info: dict, env_num: int):
        for k in info.keys():
            info_array = self.info.get(k, [None for _ in range(self.num_envs)])
            info_array[env_num] = info[k]
            self.info[k] = info_array

    def get_info(self) -> dict:
        return self.info


class StrategiesEnum(Enum):
    classic: str = "classic"
    brax: str = "brax"
