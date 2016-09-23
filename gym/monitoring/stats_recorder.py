import json
import os
import time

from gym import error
from gym.utils import atomic_write

class StatsRecorder(object):
    def __init__(self, directory, file_prefix):
        self.initial_reset_timestamp = None
        self.directory = directory
        self.file_prefix = file_prefix
        self.episode_lengths = []
        self.episode_rewards = []
        self.episode_types = [] # experimental addition
        self._type = 't'
        self.timestamps = []
        self.steps = None
        self.rewards = None

        self.done = None
        self.closed = False

        filename = '{}.stats.json'.format(self.file_prefix)
        self.path = os.path.join(self.directory, filename)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        if type not in ['t', 'e']:
            raise error.Error('Invalid episode type {}: must be t for training or e for evaluation', type)
        self._type = type

    def before_step(self, action):
        assert not self.closed

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
        assert not self.closed

        self.done = False
        if self.initial_reset_timestamp is None:
            self.initial_reset_timestamp = time.time()

    def after_reset(self, observation):
        self.save_complete()
        self.steps = 0
        self.rewards = 0
        # We write the type at the beginning of the episode. If a user
        # changes the type, it's more natural for it to apply next
        # time the user calls reset().
        self.episode_types.append(self._type)

    def save_complete(self):
        if self.steps is not None:
            self.episode_lengths.append(self.steps)
            self.episode_rewards.append(self.rewards)
            self.timestamps.append(time.time())

    def close(self):
        self.save_complete()
        self.flush()
        self.closed = True

    def flush(self):
        if self.closed:
            return

        with atomic_write.atomic_write(self.path) as f:
            json.dump({
                'initial_reset_timestamp': self.initial_reset_timestamp,
                'timestamps': self.timestamps,
                'episode_lengths': self.episode_lengths,
                'episode_rewards': self.episode_rewards,
                'episode_types': self.episode_types,
            }, f)
