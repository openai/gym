from typing import List, Optional, Union

import numpy as np

import gym
from gym.logger import deprecation
from gym.vector.utils.spaces import batch_space

__all__ = ["VectorEnv"]


class VectorEnv(gym.Env):
    r"""Base class for vectorized environments. Runs multiple independent copies of the
    same environment in parallel. This is not the same as 1 environment that has multiple
    sub components, but it is many copies of the same base env.

    Each observation returned from vectorized environment is a batch of observations
    for each parallel environment. And :meth:`step` is also expected to receive a batch of
    actions for each parallel environment.

    .. note::

        All parallel environments should share the identical observation and action spaces.
        In other words, a vector of multiple different environments is not supported.

    Parameters
    ----------
    num_envs : int
        Number of environments in the vectorized environment.

    observation_space : :class:`gym.spaces.Space`
        Observation space of a single environment.

    action_space : :class:`gym.spaces.Space`
        Action space of a single environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.is_vector_env = True
        self.observation_space = batch_space(observation_space, n=num_envs)
        self.action_space = batch_space(action_space, n=num_envs)

        self.closed = False
        self.viewer = None

        # The observation and action spaces of a single environment are
        # kept in separate properties
        self.single_observation_space = observation_space
        self.single_action_space = action_space

    def reset_async(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        pass

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        raise NotImplementedError()

    def reset(
        self,
        *,
        seed: Optional[Union[int, List[int]]] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        r"""Reset all parallel environments and return a batch of initial observations.

        Returns
        -------
        observations : element of :attr:`observation_space`
            A batch of observations from the vectorized environment.
        """
        self.reset_async(seed=seed, return_info=return_info, options=options)
        return self.reset_wait(seed=seed, return_info=return_info, options=options)

    def step_async(self, actions):
        pass

    def step_wait(self, **kwargs):
        raise NotImplementedError()

    def step(self, actions):
        r"""Take an action for each parallel environment.

        Parameters
        ----------
        actions : element of :attr:`action_space`
            Batch of actions.

        Returns
        -------
        observations : element of :attr:`observation_space`
            A batch of observations from the vectorized environment.

        rewards : :obj:`np.ndarray`, dtype :obj:`np.float_`
            A vector of rewards from the vectorized environment.

        dones : :obj:`np.ndarray`, dtype :obj:`np.bool_`
            A vector whose entries indicate whether the episode has ended.

        infos : dict
            A dict of auxiliary diagnostic information. Each `key` of the dict
            is an array of length `num_envs` where each index of the array
            represents the info coming from the i-th vectorized environment.
            Each `key` is paired with a `_key` representing whether or not the i-th
            environment has data or not.
        """

        self.step_async(actions)
        return self.step_wait()

    def call_async(self, name, *args, **kwargs):
        pass

    def call_wait(self, **kwargs):
        raise NotImplementedError()

    def call(self, name, *args, **kwargs):
        """Call a method, or get a property, from each parallel environment.

        Parameters
        ----------
        name : string
            Name of the method or property to call.

        *args
            Arguments to apply to the method call.

        **kwargs
            Keywoard arguments to apply to the method call.

        Returns
        -------
        results : list
            List of the results of the individual calls to the method or
            property for each environment.
        """
        self.call_async(name, *args, **kwargs)
        return self.call_wait()

    def get_attr(self, name):
        """Get a property from each parallel environment.

        Parameters
        ----------
        name : string
            Name of the property to be get from each individual environment.
        """
        return self.call(name)

    def set_attr(self, name, values):
        """Set a property in each parallel environment.

        Parameters
        ----------
        name : string
            Name of the property to be set in each individual environment.

        values : list, tuple, or object
            Values of the property to be set to. If `values` is a list or
            tuple, then it corresponds to the values for each individual
            environment, otherwise a single value is set for all environments.
        """
        raise NotImplementedError()

    def close_extras(self, **kwargs):
        r"""Clean up the extra resources e.g. beyond what's in this base class."""
        pass

    def close(self, **kwargs):
        r"""Close all parallel environments and release resources.

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

    def seed(self, seed=None):
        """Set the random seed in all parallel environments.

        Parameters
        ----------
        seed : list of int, or int, optional
            Random seed for each parallel environment. If ``seed`` is a list of
            length ``num_envs``, then the items of the list are chosen as random
            seeds. If ``seed`` is an int, then each parallel environment uses the random
            seed ``seed + n``, where ``n`` is the index of the parallel environment
            (between ``0`` and ``num_envs - 1``).
        """
        deprecation(
            "Function `env.seed(seed)` is marked as deprecated and will be removed in the future. "
            "Please use `env.reset(seed=seed) instead in VectorEnvs."
        )

    def _add_info(self, infos: dict, info: dict, env_num: int) -> dict:
        """Add env info to the info dictionary of the vectorized environment.

        Given the `info` of a single environment add it to the `infos` dictionary
        which represents all the infos of the vectorized environment.
        Every `key` of `info` is paired with a boolean mask `_key` representing
        whether or not the i-indexed environment has this `info`.

        Args:
            infos (dict): the infos of the vectorized environment
            info (dict): the info coming from the single environment
            env_num (int): the index of the single environment

        Returns:
            infos (dict): the (updated) infos of the vectorized environment

        """
        for k in info.keys():
            if k not in infos:
                info_array, array_mask = self._init_info_arrays(type(info[k]))
            else:
                info_array, array_mask = infos[k], infos[f"_{k}"]

            info_array[env_num], array_mask[env_num] = info[k], True
            infos[k], infos[f"_{k}"] = info_array, array_mask
        return infos

    def _init_info_arrays(self, dtype: type) -> np.ndarray:
        if dtype in [int, float, bool] or issubclass(dtype, np.number):
            array = np.zeros(self.num_envs, dtype=dtype)
        else:
            array = np.zeros(self.num_envs, dtype=object)
            array[:] = None
        array_mask = np.zeros(self.num_envs, dtype=bool)
        return array, array_mask

    def __del__(self):
        if not getattr(self, "closed", True):
            self.close()

    def __repr__(self):
        if self.spec is None:
            return f"{self.__class__.__name__}({self.num_envs})"
        else:
            return f"{self.__class__.__name__}({self.spec.id}, {self.num_envs})"


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
    def reset_async(self, **kwargs):
        return self.env.reset_async(**kwargs)

    def reset_wait(self, **kwargs):
        return self.env.reset_wait(**kwargs)

    def step_async(self, actions):
        return self.env.step_async(actions)

    def step_wait(self):
        return self.env.step_wait()

    def close(self, **kwargs):
        return self.env.close(**kwargs)

    def close_extras(self, **kwargs):
        return self.env.close_extras(**kwargs)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def call(self, name, *args, **kwargs):
        return self.env.call(name, *args, **kwargs)

    def set_attr(self, name, values):
        return self.env.set_attr(name, values)

    # implicitly forward all other methods and attributes to self.env
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __repr__(self):
        return f"<{self.__class__.__name__}, {self.env}>"

    def __del__(self):
        self.env.__del__()
