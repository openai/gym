from typing import List, Union, Optional

import numpy as np
from copy import deepcopy

from gym import logger
from gym.logger import warn
from gym.vector.vector_env import VectorEnv
from gym.vector.utils import concatenate, iterate, create_empty_array

__all__ = ["SyncVectorEnv"]


class SyncVectorEnv(VectorEnv):
    """Vectorized environment that serially runs multiple environments.

    Parameters
    ----------
    env_fns : iterable of callable
        Functions that create the environments.

    observation_space : :class:`gym.spaces.Space`, optional
        Observation space of a single environment. If ``None``, then the
        observation space of the first environment is taken.

    action_space : :class:`gym.spaces.Space`, optional
        Action space of a single environment. If ``None``, then the action space
        of the first environment is taken.

    copy : bool
        If ``True``, then the :meth:`reset` and :meth:`step` methods return a
        copy of the observations.

    Raises
    ------
    RuntimeError
        If the observation space of some sub-environment does not match
        :obj:`observation_space` (or, by default, the observation space of
        the first sub-environment).

    Example
    -------

    .. code-block::

        >>> env = gym.vector.SyncVectorEnv([
        ...     lambda: gym.make("Pendulum-v0", g=9.81),
        ...     lambda: gym.make("Pendulum-v0", g=1.62)
        ... ])
        >>> env.reset()
        array([[-0.8286432 ,  0.5597771 ,  0.90249056],
               [-0.85009176,  0.5266346 ,  0.60007906]], dtype=float32)
    """

    def __init__(self, env_fns, observation_space=None, action_space=None, copy=True):
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy
        self.metadata = self.envs[0].metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        super().__init__(
            num_envs=len(env_fns),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_spaces()
        self.observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._dones = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None

    def seed(self, seed=None):
        super().seed(seed=seed)
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        for env, single_seed in zip(self.envs, seed):
            env.seed(single_seed)

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        self._dones[:] = False
        observations = []
        data_list = []
        for env, single_seed in zip(self.envs, seed):

            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options
            if return_info == True:
                kwargs["return_info"] = return_info

            if not return_info:
                observation = env.reset(**kwargs)
                observations.append(observation)
            else:
                observation, data = env.reset(**kwargs)
                observations.append(observation)
                data_list.append(data)

        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )
        if not return_info:
            return deepcopy(self.observations) if self.copy else self.observations
        else:
            return (
                deepcopy(self.observations) if self.copy else self.observations
            ), data_list

    def step_async(self, actions):
        self._actions = iterate(self.action_space, actions)

    def step_wait(self):
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(action)
            if self._dones[i]:
                info["terminal_observation"] = observation
                observation = env.reset()
            observations.append(observation)
            infos.append(info)
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._dones),
            infos,
        )

    def call(self, name, *args, **kwargs):
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def set_attr(self, name, values):
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            setattr(env, name, value)

    def close_extras(self, **kwargs):
        """Close the environments."""
        [env.close() for env in self.envs]

    def _check_spaces(self):
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                raise RuntimeError(
                    "Some environments have an observation space different from "
                    f"`{self.single_observation_space}`. In order to batch observations, "
                    "the observation spaces from all environments must be equal."
                )

            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    "Some environments have an action space different from "
                    f"`{self.single_action_space}`. In order to batch actions, the "
                    "action spaces from all environments must be equal."
                )

        else:
            return True
