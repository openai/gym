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

    def __del__(self):
        if hasattr(self, 'closed'):
            if not self.closed:
                self.close()


class VectorEnvWrapper(VectorEnv):
    r"""Wraps the vectorized environment to allow a modular transformation. 
    
    This class is the base class for all wrappers for vectorized environments. The subclass
    could override some methods to change the behavior of the original vectorized environment
    without touching the original code. 
    
    .. note::
    
        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.
    
    """
    def __init__(self, env):
        assert isinstance(env, VectorEnv)
        self.env = env
        self.metadata = env.metadata
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.single_observation_space = env.single_observation_space
        self.single_action_space = env.single_action_space
        self.reward_range = env.reward_range

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    @property
    def spec(self):
        return self.env.spec

    @classmethod
    def class_name(cls):
        return cls.__name__

    def seed(self, seeds=None):
        return self.env.seed(seeds)

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def get_images(self):
        return self.env.get_images()

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self, **kwargs):
        return self.env.close(**kwargs)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __repr__(self):
        return '<{}, {}>'.format(self.__class__.__name__, self.env)
