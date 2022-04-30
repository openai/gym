import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

import numpy as np

import gym
from gym.error import InvalidInfoStrategy, NoMatchingInfoStrategy
from gym.vector.utils import InfoStrategiesEnum


class StatstInfoStrategy(ABC):
    """Interface for implementing a info
    processing strategy for RecordEpisodeStatistics
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
    def add_episode_statistics(self, infos: dict, env_num: int):
        """Add the episode statistics to the info"""

    @abstractmethod
    def get_info(self):
        """Return the info for the
        vectorized env
        """


class ClassicStatsInfoStrategy(StatstInfoStrategy):
    def __init__(self, num_envs: int):
        self.info = {}

    def add_info(self, infos: dict, env_num: int):
        self.info = {**self.info, **infos}

    def add_episode_statistics(self, infos: dict, env_num: int):
        self.info = {**self.info, **infos}

    def get_info(self):
        return self.info


class ClassicVecEnvStatsInfoStrategy(StatstInfoStrategy):
    def __init__(self, num_envs: int):
        self.info = [{} for _ in range(num_envs)]

    def add_info(self, infos: list, env_num: int):
        self.info[env_num] = {**self.info[env_num], **infos[env_num]}

    def add_episode_statistics(self, infos: dict, env_num: int):
        self.info[env_num] = {**self.info[env_num], **infos}

    def get_info(self):
        return tuple(self.info)


class BraxVecEnvStatsInfoStrategy(StatstInfoStrategy):
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.info = {}

    def add_info(self, info: dict, env_num: int):
        self.info = {**self.info, **info}

    def add_episode_statistics(self, info: dict, env_num: int):
        episode_info = info["episode"]
        self.info["episode"] = self.info.get("episode", {})
        for k in episode_info.keys():
            info_array = self.info["episode"].get(k, np.zeros(self.num_envs))
            info_array[env_num] = episode_info[k]
            self.info["episode"][k] = info_array

    def get_info(self):
        return self.info


class StatsInfoStrategyFactory:
    _strategies = {
        InfoStrategiesEnum.classic.value: ClassicVecEnvStatsInfoStrategy,
        InfoStrategiesEnum.brax.value: BraxVecEnvStatsInfoStrategy,
    }

    def get_stats_info_strategy(wrapped_env_strategy: str):
        if wrapped_env_strategy not in StatsInfoStrategyFactory._strategies:
            raise NoMatchingInfoStrategy(
                f"Wrapped environment has an info format of type {wrapped_env_strategy} which is not a processable format by this wrapper. Please use one in {list(StatsInfoStrategyFactory._strategies.keys())}"
            )
        return StatsInfoStrategyFactory._strategies[wrapped_env_strategy]

    def add_info_strategy(name: str, strategy: StatstInfoStrategy) -> None:
        if not issubclass(strategy, StatstInfoStrategy):
            raise InvalidInfoStrategy(
                "The strategy need to subclass the Abstract Base Class `StatstInfoStrategy`"
            )
        StatsInfoStrategyFactory._strategies[name] = strategy


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.episode_returns = None
        self.episode_lengths = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.stats_info_strategy = StatsInfoStrategyFactory.get_stats_info_strategy(
                self.env.info_format
            )
        else:
            self.stats_info_strategy = ClassicStatsInfoStrategy

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        infos_processor = self.stats_info_strategy(self.num_envs)
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        if not self.is_vector_env:
            dones = [dones]
        dones = list(dones)

        for i in range(len(dones)):
            if dones[i]:
                infos_processor.add_info(infos, i)
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "episode": {
                        "r": episode_return,
                        "l": episode_length,
                        "t": round(time.perf_counter() - self.t0, 6),
                    }
                }
                infos_processor.add_episode_statistics(episode_info, i)
                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        return (
            observations,
            rewards,
            dones if self.is_vector_env else dones[0],
            infos_processor.get_info(),
        )
