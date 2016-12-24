from gym import monitoring
from gym import Wrapper

import logging

logger = logging.getLogger(__name__)

def Monitor(directory, video_callable=None, force=False, resume=False,
            write_upon_reset=False, uid=None, mode=None):
    class Monitor(Wrapper):
        def __init__(self, env):
            super(Monitor, self).__init__(env)
            self._monitor = monitoring.MonitorManager(env)
            self._monitor.start(directory, video_callable, force, resume,
                                write_upon_reset, uid, mode)

        def _step(self, action):
            self._monitor._before_step(action)
            observation, reward, done, info = self.env.step(action)
            done = self._monitor._after_step(observation, reward, done, info)

            return observation, reward, done, info

        def _reset(self):
            self._monitor._before_reset()
            observation = self.env.reset()
            self._monitor._after_reset(observation)

            return observation

        def _close(self):
            super(Monitor, self)._close()

            # _monitor will not be set if super(Monitor, self).__init__ raises, this check prevents a confusing error message
            if getattr(self, '_monitor', None):
                self._monitor.close()

        def set_monitor_mode(self, mode):
            logger.info("Setting the monitor mode is deprecated and will be removed soon")
            self._monitor._set_mode(mode)
    return Monitor
