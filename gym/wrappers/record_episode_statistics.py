"""Wrapper that tracks the cumulative rewards and episode lengths."""
import time
from collections import deque

import numpy as np

import gym


class ClassicStatsInfo:
    """Manage episode statistics."""

    def __init__(self, num_envs: int):
        """Classic EpisodeStatics info.

        Args:
            num_envs (int): number of environments.
        """
        self.info = {}

    def add_info(self, infos: dict, env_num: int):
        """Add info.

        Args:
            infos (dict): info dict of the environment.
            env_num (int): environment number.
        """
        self.info = {**self.info, **infos}

    def add_episode_statistics(self, infos: dict, env_num: int):
        """Add episode statistics.

        Args:
            infos (dict): info dict of the environment.
            env_num (int): env number.
        """
        self.info = {**self.info, **infos}

    def get_info(self):
        """Return info."""
        return self.info


class BraxVecEnvStatsInfo:
    """Manage episode statistics in the Brax format for vectorized envs."""

    def __init__(self, num_envs: int):
        """Brax-style episode statistics.

        Args:
            num_envs (int): number of environments.
        """
        self.num_envs = num_envs
        self.info = {}

    def add_info(self, info: dict, env_num: int):
        """Add info.

        Args:
            info (dict): info dict of the environment.
            env_num (int): environment number.
        """
        self.info = {**self.info, **info}

    def add_episode_statistics(self, info: dict, env_num: int):
        """Add episode statistics.

        Add statistics coming from the vectorized environment.

        Args:
            info (dict): info dict of the environment.
            env_num (int): env number of the vectorized environments.
        """
        episode_info = info["episode"]

        self.info["episode"] = self.info.get("episode", {})

        self.info["_episode"] = self.info.get(
            "_episode", np.zeros(self.num_envs, dtype=bool)
        )
        self.info["_episode"][env_num] = True

        for k in episode_info.keys():
            info_array = self.info["episode"].get(k, np.zeros(self.num_envs))
            info_array[env_num] = episode_info[k]
            self.info["episode"][k] = info_array

    def get_info(self):
        """Returns info."""
        return self.info


class RecordEpisodeStatistics(gym.Wrapper):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``. After the completion
    of an episode, ``info`` will look like this::

        >>> info = {
        ...     ...
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since instantiation of wrapper>"
        ...     },
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes
    """

    def __init__(self, env: gym.Env, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
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
            self.stats_info_processor = BraxVecEnvStatsInfo
        else:
            self.stats_info_processor = ClassicStatsInfo

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        infos_processor = self.stats_info_processor(self.num_envs)
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
