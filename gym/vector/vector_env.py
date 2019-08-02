import gym
from gym.spaces import Tuple
from gym.vector.utils.spaces import batch_space

__all__ = ['VectorEnv']


class VectorEnv(gym.Env):
    """Base class for vectorized environments.

    Parameters
    ----------
    num_envs : int
        Number of environments in the vectorized environment.

    observation_space : `gym.spaces.Space` instance
        Observation space of a single environment.

    action_space : `gym.spaces.Space` instance
        Action space of a single environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        super(VectorEnv, self).__init__()
        self.num_envs = num_envs
        self.observation_space = batch_space(observation_space, n=num_envs)
        self.action_space = Tuple((action_space,) * num_envs)

        self.closed = False
        self.viewer = None

        # The observation and action spaces of a single environment are
        # kept in separate properties
        self.single_observation_space = observation_space
        self.single_action_space = action_space

    def reset_async(self):
        pass

    def reset_wait(self, **kwargs):
        raise NotImplementedError()

    def reset(self):
        self.reset_async()
        return self.reset_wait()

    def step_async(self, actions):
        pass

    def step_wait(self, **kwargs):
        raise NotImplementedError()

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close_extras(self, **kwargs):
        r"""Clean up the extra resources e.g. beyond what's in this base class. """
        raise NotImplementedError()

    def close(self, **kwargs):
        r"""Close all sub-environments and release resources.
        
        It also closes all the existing image viewers, then calls :meth:`close_extras` and set
        :attr:`closed` as ``True``. 
        
        .. warning::
        
            This function itself does not close the environments, it should be handled
            in :meth:`close_extras`. This is generic for both synchronous and asynchronous 
            vectorized environments. 
        
        .. note::
        
            This will be automatically called when garbage collected or program exited. 
            
        """
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras(**kwargs)
        self.closed = True

    def __del__(self):
        if hasattr(self, 'closed'):
            if not self.closed:
                self.close(terminate=True)
