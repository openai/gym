import logging

from gym import Wrapper
from gym import error
from gym import monitoring
from gym.wrappers.frame_skipping import SkipWrapper

logger = logging.getLogger(__name__)

class Monitored(Wrapper):
    def __init__(self, env):
        super(Monitored, self).__init__(env)
        self._monitor = monitoring.Monitor(env)

    def _configure(self, *args, monitor_config=None, **kwargs):
        super(Monitored, self)._configure(*args, **kwargs)

        if not monitor_config:
            monitor_config = {}

        if not 'directory' in monitor_config:
            raise error.Error("Required argument 'directory' not found in monitor_config")

        self._monitor.start(**monitor_config)

    def _step(self, action):
        self._monitor._before_step(action)
        observation, reward, done, info = self.env.step(action)
        self._monitor._after_step(observation, reward, done, info)

        return observation, reward, done, info

    def _reset(self):
        self._monitor._before_reset()
        observation= self.env.reset()
        self._monitor._after_reset(observation)

        return observation

    def _close(self):
        super(Monitored, self)._close()
        self._monitor.close()

    def set_monitor_mode(self, mode):
        logger.info("Setting the monitor mode is deprecated and will be removed soon")
        self._monitor._set_mode(mode)
