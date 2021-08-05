import time
from collections import deque
import numpy as np
import gym


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.env_is_vec = isinstance(env, gym.vector.VectorEnv)
        self.num_envs = getattr(env, "num_envs", 1)
        self.t0 = (
            time.time()
        )  # TODO: use perf_counter when gym removes Python 2 support
        self.episode_count = 0
        self.episode_returns = None
        self.episode_lengths = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observations = super(RecordEpisodeStatistics, self).reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super(RecordEpisodeStatistics, self).step(
            action
        )
        self.episode_returns += rewards
        self.episode_lengths += 1
        if not self.env_is_vec:
            infos = [infos]
            dones = [dones]
        for i in range(len(dones)):
            if dones[i]:
                infos[i] = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return,
                    "l": episode_length,
                    "t": round(time.time() - self.t0, 6),
                }
                infos[i]["episode"] = episode_info
                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        return (
            observations,
            rewards,
            dones if self.env_is_vec else dones[0],
            infos if self.env_is_vec else infos[0],
        )
