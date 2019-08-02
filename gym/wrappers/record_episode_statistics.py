import time
from collections import deque

import gym


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.t0 = time.perf_counter()
        self.episode_return = 0.0
        self.episode_horizon = 0
        self.return_queue = deque(maxlen=deque_size)
        self.horizon_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.episode_return = 0.0
        self.episode_horizon = 0
        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.episode_return += reward
        self.episode_horizon += 1
        if done:
            info['episode'] = {'return': self.episode_return, 
                               'horizon': self.episode_horizon, 
                               'time': round(time.perf_counter() - self.t0, 4)}
            self.return_queue.append(self.episode_return)
            self.horizon_queue.append(self.episode_horizon)
            self.episode_return = 0.0
            self.episode_horizon = 0
        return observation, reward, done, info
