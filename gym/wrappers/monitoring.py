from gym import monitoring
from gym import Wrapper

class Monitored(Wrapper):
    """Attaches a monitor to the wrapped environment that records stats and
    video. When this wrapper is created, it creates a monitor and calls start()
    on it. Calling close() on the wrapper env will also close() the monitor.

    For finer-grained control, the monitor can be interacted with directly using
    the 'monitor' attribute.
    """
    def __init__(self, env, *start_args, **start_kwargs):
        """Monitor is started immediately when this wrapper is created, using
        the provided args. See monitoring.Monitor.start for documentation of 
        start parameters."""
        self.monitor = monitoring.Monitor(env)
        super(Monitored, self).__init__(env)
        self.monitor.start(*start_args, **start_kwargs)

    def _step(self, action):
        self.monitor._before_step(action)
        observation, reward, done, info = self.env.step(action)
        done = self.monitor._after_step(observation, reward, done, info)
        return observation, reward, done, info

    def _reset(self):
        self.monitor._before_reset()
        observation = self.env.reset()
        self.monitor._after_reset(observation)
        return observation

    def _close(self):
        self.monitor.close()
        self.env.close()
