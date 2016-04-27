import json
import os
import time

from gym import error

class StatsRecorder(object):
    def __init__(self, directory, file_prefix):
        self.initial_reset_timestamp = None
        self.directory = directory
        self.file_prefix = file_prefix
        self.episode_lengths = []
        self.episode_rewards = []
        self.timestamps = []
        self.steps = None
        self.rewards = None

        self.done = None

    def before_step(self, action):
        if self.done:
            raise error.ResetNeeded("Trying to step environment which is currently done. While the monitor is active, you cannot step beyond the end of an episode. Call 'env.reset()' to start the next episode.")
        elif self.steps is None:
            raise error.ResetNeeded("Trying to step an environment before reset. While the monitor is active, you must call 'env.reset()' before taking an initial step.")

    def after_step(self, observation, reward, done, info):
        self.steps += 1
        self.rewards += reward
        if done:
            self.done = True

    def before_reset(self):
        self.done = False
        if self.initial_reset_timestamp is None:
            self.initial_reset_timestamp = time.time()

    def after_reset(self, observation):
        self.flush()

    def flush(self):
        if self.steps is not None:
            self.episode_lengths.append(self.steps)
            self.episode_rewards.append(self.rewards)
            self.timestamps.append(time.time())
        self.steps = 0
        self.rewards = 0

    def close(self):
        self.flush()

        filename = '{}.{}.stats.json'.format(self.file_prefix, os.getpid())
        path = os.path.join(self.directory, filename)
        with open(path, 'w') as f:
            json.dump({
                'initial_reset_timestamp': self.initial_reset_timestamp,
                'timestamps': self.timestamps,
                'episode_lengths': self.episode_lengths,
                'episode_rewards': self.episode_rewards,
            }, f)
        return path
