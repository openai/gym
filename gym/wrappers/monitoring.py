import gym
from gym import monitoring
from gym import Wrapper
from gym.wrappers.time_limit import TimeLimit
from gym import error

import logging

logger = logging.getLogger(__name__)

class _Monitor(Wrapper):
    def __init__(self, env, directory, video_callable=None, force=False, resume=False,
                 write_upon_reset=False, uid=None, mode=None):
        super(_Monitor, self).__init__(env)
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
        super(_Monitor, self)._close()

        # _monitor will not be set if super(Monitor, self).__init__ raises, this check prevents a confusing error message
        if getattr(self, '_monitor', None):
            self._monitor.close()

    def set_monitor_mode(self, mode):
        logger.info("Setting the monitor mode is deprecated and will be removed soon")
        self._monitor._set_mode(mode)

def Monitor(env=None, directory=None, video_callable=None, force=False, resume=False,
            write_upon_reset=False, uid=None, mode=None):
    if not isinstance(env, gym.Env):
        raise error.Error("Monitor decorator syntax is deprecated as of 12/28/2016. Replace your call to `env = gym.wrappers.Monitor(directory)(env)` with `env = gym.wrappers.Monitor(env, directory)`")

    return _Monitor(TimeLimit(env), directory, video_callable, force, resume,
                    write_upon_reset, uid, mode)
