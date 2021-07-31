import gym
from gym.spaces import Tuple
from gym.vector.utils.spaces import batch_space

__all__ = ["VectorEnv"]


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

    def close_extras(self, **kwargs):
        r"""Clean up the extra resources e.g. beyond what's in this base class."""
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

    def seed(self, seeds=None):
        """
        Parameters
        ----------
        seeds : list of int, or int, optional
            Random seed for each individual environment. If `seeds` is a list of
            length `num_envs`, then the items of the list are chosen as random
            seeds. If `seeds` is an int, then each environment uses the random
            seed `seeds + n`, where `n` is the index of the environment (between
            `0` and `num_envs - 1`).
        """
        pass

    def __del__(self):
        if not getattr(self, "closed", True):
            self.close(terminate=True)

    def __repr__(self):
        if self.spec is None:
            return "{}({})".format(self.__class__.__name__, self.num_envs)
        else:
            return "{}({}, {})".format(
                self.__class__.__name__, self.spec.id, self.num_envs
            )


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

    # explicitly forward the methods defined in VectorEnv
    # to self.env (instead of the base class)
    def reset_async(self):
        return self.env.reset_async()

    def reset_wait(self):
        return self.env.reset_wait()

    def step_async(self, actions):
        return self.env.step_async(actions)

    def step_wait(self):
        return self.env.step_wait()

    def close(self, **kwargs):
        return self.env.close(**kwargs)

    def close_extras(self, **kwargs):
        return self.env.close_extras(**kwargs)

    def seed(self, seeds=None):
        return self.env.seed(seeds)

    # implicitly forward all other methods and attributes to self.env
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __repr__(self):
        return "<{}, {}>".format(self.__class__.__name__, self.env)
