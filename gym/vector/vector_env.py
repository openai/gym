import gym
from gym.spaces import Tuple
from gym.vector.utils.spaces import batch_space

__all__ = ['VectorEnv']


class VectorEnv(gym.Env):
    r"""Base class for vectorized environments.

    Each observation returned from vectorized environment is a batch of observations 
    for each sub-environment. And :meth:`step` is also expected to receive a batch of 
    actions for each sub-environment.
    
    .. note::
    
        All sub-environments should share the identical observation and action spaces.
        In other words, a vector of multiple different environments is not supported. 

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
        r"""Reset all sub-environments and return a batch of initial observations.
        
        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.
        """
        self.reset_async()
        return self.reset_wait()

    def step_async(self, actions):
        pass

    def step_wait(self, **kwargs):
        raise NotImplementedError()

    def step(self, actions):
        r"""Take an action for each sub-environments. 

        Parameters
        ----------
        actions : iterable of samples from `action_space`
            List of actions.

        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.
        rewards : `np.ndarray` instance (dtype `np.float_`)
            A vector of rewards from the vectorized environment.
        dones : `np.ndarray` instance (dtype `np.bool_`)
            A vector whose entries indicate whether the episode has ended.
        infos : list of dict
            A list of auxiliary diagnostic information dicts from sub-environments.
        """
        
        self.step_async(actions)
        return self.step_wait()

    def __del__(self):
        if hasattr(self, 'closed'):
            if not self.closed:
                self.close()
