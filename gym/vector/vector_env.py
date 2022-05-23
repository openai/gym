"""Base class for vectorized environments."""
from __future__ import annotations

from typing import Any, Optional, Union

import gym
from gym.logger import deprecation
from gym.vector.utils.spaces import batch_space

__all__ = ["VectorEnv"]


class VectorEnv(gym.Env):
    """Base class for vectorized environments. Runs multiple independent copies of the same environment in parallel.

    This is not the same as 1 environment that has multiple subcomponents, but it is many copies of the same base env.

    Each observation returned from vectorized environment is a batch of observations for each parallel environment.
    And :meth:`step` is also expected to receive a batch of actions for each parallel environment.

    Notes:
        All parallel environments should share the identical observation and action spaces.
        In other words, a vector of multiple different environments is not supported.
    """

    def __init__(
        self, num_envs: int, observation_space: gym.Space, action_space: gym.Space
    ):
        """Base class for vectorized environments.

        Args:
            num_envs: Number of environments in the vectorized environment.
            observation_space: Observation space of a single environment.
            action_space: Action space of a single environment.
        """
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
        seed: Optional[Union[int, list[int]]] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        """Reset the sub-environments asynchronously.

        This method will return ``None``. A call to :meth:`reset_async` should be followed by a call to :meth:`reset_wait` to retrieve the results.
        """
        pass

    def reset_wait(
        self,
        seed: Optional[Union[int, list[int]]] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        """Retrieves the results of a :meth:`reset_async` call.

        A call to this method must always be preceded by a call to :meth:`reset_async`.
        """
        raise NotImplementedError()

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        """Reset all parallel environments and return a batch of initial observations.

        Args:
            seed: The environment reset seeds
            return_info: If to return the info
            options: If to return the options

        Returns:
            A batch of observations from the vectorized environment.
        """
        self.reset_async(seed=seed, return_info=return_info, options=options)
        return self.reset_wait(seed=seed, return_info=return_info, options=options)

    def step_async(self, actions):
        """Asynchronously performs steps in the sub-environments.

        The results can be retrieved via a call to :meth:`step_wait`.
        """
        pass

    def step_wait(self, **kwargs):
        """Retrieves the results of a :meth:`step_async` call.

        A call to this method must always be preceded by a call to :meth:`step_async`.
        """
        raise NotImplementedError()

    def step(self, actions):
        """Take an action for each parallel environment.

        Args:
            actions: element of :attr:`action_space` Batch of actions.

        Returns:
            Batch of observations, rewards, done and infos
        """
        self.step_async(actions)
        return self.step_wait()

    def call_async(self, name, *args, **kwargs):
        """Calls a method name for each parallel environment asynchronously."""
        pass

    def call_wait(self, **kwargs):
        """After calling a method in :meth:`call_async`, this function collects the results."""
        raise NotImplementedError()

    def call(self, name: str, *args, **kwargs) -> list[Any]:
        """Call a method, or get a property, from each parallel environment.

        Args:
            name (str): Name of the method or property to call.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Returns:
            List of the results of the individual calls to the method or property for each environment.
        """
        self.call_async(name, *args, **kwargs)
        return self.call_wait()

    def get_attr(self, name: str):
        """Get a property from each parallel environment.

        Args:
            name (str): Name of the property to be get from each individual environment.

        Returns:
            The property with name
        """
        return self.call(name)

    def set_attr(self, name: str, values: Union[list, tuple, object]):
        """Set a property in each sub-environment.

        Args:
            name (str): Name of the property to be set in each individual environment.
            values (list, tuple, or object): Values of the property to be set to. If `values` is a list or
                tuple, then it corresponds to the values for each individual environment, otherwise a single value
                is set for all environments.
        """
        raise NotImplementedError()

    def close_extras(self, **kwargs):
        """Clean up the extra resources e.g. beyond what's in this base class."""
        pass

    def close(self, **kwargs):
        """Close all parallel environments and release resources.

        It also closes all the existing image viewers, then calls :meth:`close_extras` and set
        :attr:`closed` as ``True``.

        Warnings:
            This function itself does not close the environments, it should be handled
            in :meth:`close_extras`. This is generic for both synchronous and asynchronous
            vectorized environments.

        Notes:
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

        Args:
            seed: Random seed for each parallel environment. If ``seed`` is a list of
                length ``num_envs``, then the items of the list are chosen as random
                seeds. If ``seed`` is an int, then each parallel environment uses the random
                seed ``seed + n``, where ``n`` is the index of the parallel environment
                (between ``0`` and ``num_envs - 1``).
        """
        deprecation(
            "Function `env.seed(seed)` is marked as deprecated and will be removed in the future. "
            "Please use `env.reset(seed=seed) instead in VectorEnvs."
        )

    def __del__(self):
        """Closes the vector environment."""
        if not getattr(self, "closed", True):
            self.close()

    def __repr__(self):
        """Returns a string representation of the vector environment using the class name, number of environments and environment spec id."""
        if self.spec is None:
            return f"{self.__class__.__name__}({self.num_envs})"
        else:
            return f"{self.__class__.__name__}({self.spec.id}, {self.num_envs})"


class VectorEnvWrapper(VectorEnv):
    """Wraps the vectorized environment to allow a modular transformation.

    This class is the base class for all wrappers for vectorized environments. The subclass
    could override some methods to change the behavior of the original vectorized environment
    without touching the original code.

    Notes:
        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.
    """

    def __init__(self, env: VectorEnv):
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
