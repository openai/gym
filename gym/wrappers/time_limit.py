import time

from gym import Wrapper

import logging

logger = logging.getLogger(__name__)

DEFAULT_MAX_EPISODE_SECONDS = 20 * 60.  # Default to 20 minutes if there is no explicit limit

class TimeLimit(Wrapper):
    def _configure(self, **kwargs):
        super(TimeLimit, self)._configure(**kwargs)
        self._max_episode_seconds = self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_seconds', None)
        self._max_episode_steps = self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps', None)

        if self._max_episode_seconds is None and self._max_episode_steps is None:
            self._max_episode_seconds = DEFAULT_MAX_EPISODE_SECONDS

        self._elapsed_steps = 0
        self._episode_started_at = None

    @property
    def _elapsed_seconds(self):
        return time.time() - self._episode_started_at

    def _past_limit(self):
        """Return true if we are past our limit"""
        if self._max_episode_steps is not None and self._max_episode_steps <= self._elapsed_steps:
            logger.debug("Env has passed the step limit defined by TimeLimit.")
            return True

        if self._max_episode_seconds is not None and self._max_episode_seconds <= self._elapsed_seconds:
            logger.debug("Env has passed the seconds limit defined by TimeLimit.")
            return True

        return False

    def _step(self, action):
        assert self._episode_started_at is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._past_limit():
            # TODO(jie) we are resetting and discarding the observation here.
            # This _should_ be fine since you can always call reset() again to
            # get a new, freshly initialized observation, but it would be better
            # to clean this up.
            _ = self.reset()  # Force a reset, discard the observation
            done = True  # Force a done = True

        return observation, reward, done, info

    def _reset(self):
        self._episode_started_at = time.time()
        self._elapsed_steps = 0
        return self.env.reset()
