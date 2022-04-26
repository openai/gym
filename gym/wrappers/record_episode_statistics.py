import time
from collections import deque
from typing import Optional

import numpy as np

import gym
from gym.error import NoMatchingInfoStrategy
from gym.vector.utils import InfoStrategiesEnum


class ClassicStatsInfoStrategy:
    def __init__(self, num_envs: int):
        self.info = {}

    def add_info(self, infos: dict, env_num: int):
        self.info = {**self.info, **infos}

    def add_episode_statistics(self, infos: dict, env_num: int):
        self.info = {**self.info, **infos}

    def get_info(self):
        return self.info


class ClassicVecEnvStatsInfoStrategy:
    def __init__(self, num_envs: int):
        self.info = [{} for _ in range(num_envs)]

    def add_info(self, infos: list, env_num: int):
        self.info[env_num] = {**self.info[env_num], **infos[env_num]}

    def add_episode_statistics(self, infos: dict, env_num: int):
        self.info[env_num] = {**self.info[env_num], **infos}

    def get_info(self):
        return tuple(self.info)


class BraxVecEnvStatsInfoStrategy:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.info = {}

    def add_info(self, info: dict, env_num: int):
        self.info = {**self.info, **info}

    def add_episode_statistics(self, info: dict, env_num: int):
        episode_info = info["episode"]
        self.info["episode"] = self.info.get("episode", {})
        for k in episode_info.keys():
            info_array = self.info["episode"].get(
                k, [None for _ in range(self.num_envs)]
            )
            info_array[env_num] = episode_info[k]
            self.info["episode"][k] = info_array

    def get_info(self):
        return self.info


def get_statistic_info_strategy(wrapped_env_strategy: str):
    strategies = {
        InfoStrategiesEnum.classic.value: ClassicVecEnvStatsInfoStrategy,
        InfoStrategiesEnum.brax.value: BraxVecEnvStatsInfoStrategy,
    }
    if wrapped_env_strategy not in strategies:
        raise NoMatchingInfoStrategy(
            "Wrapped environment has an info format of type %s which is not a processable format by this wrapper. Please use one in %s"
            % (wrapped_env_strategy, list(strategies.keys()))
        )
    return strategies[wrapped_env_strategy]


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
            self.StatsInfoStrategy = get_statistic_info_strategy(self.env.info_format)
        else:
            self.StatsInfoStrategy = ClassicStatsInfoStrategy

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        infos_processor = self.StatsInfoStrategy(self.num_envs)
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
