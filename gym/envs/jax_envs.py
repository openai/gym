import abc
import inspect
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jumpy as jp
from flax import struct

import gym
from gym import logger, spaces
from gym.core import ActType, ObsType
from gym.utils import seeding


@struct.dataclass
class JaxState:
    """Environment state.

    * state: the hidden environment state
    * obs: the agent observation to act upon
    * reward: the agent reward for a step
    * terminated: if the environment has terminated
    * truncated: if the environment has truncated
    * info: additional information for the agent or user
    """

    state: struct.dataclass
    obs: jp.ndarray
    reward: jp.ndarray
    terminate: jp.ndarray
    truncate: jp.ndarray
    info: Dict[str, Any]


class JaxEnv(gym.Env[ObsType, ActType], abc.ABC):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        reset_fn: Callable[[jax.ShapedArray, Optional[Dict[str, Any]]], JaxState],
        step_fn: Callable[
            [JaxState, jax.ShapedArray, jax.ShapedArray],
            Tuple[JaxState, jax.ShapedArray],
        ],
        jit_reset_fn: bool = True,
        jit_step_fn: bool = True,
    ):
        self.observation_space = observation_space
        self.action_space = action_space

        reset_signature = inspect.signature(reset_fn)
        if "self" in reset_signature:
            logger.warn(
                "The reset function contains the argument `self`, we recommend that the reset function is stateless and not include `self` for optimisation."
            )
        step_signature = inspect.signature(step_fn)
        if "self" in step_signature:
            logger.warn(
                "The step function contains the argument `self`, we recommend that the step function is stateless and not include `self` for optimisation."
            )

        self.internal_reset = (
            jax.jit(reset_fn, static_argnums=[1]) if jit_reset_fn else reset_fn
        )
        self.internal_step = jax.jit(step_fn) if jit_step_fn else step_fn

        self.state: Optional[JaxState] = None
        _, seed = seeding.np_random()
        self.np_random = jax.random.PRNGKey(seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        # initialise the np_random
        if seed is not None:
            self.np_random = jax.random.PRNGKey(seed)

        # Collect the initial state of the environment
        self.state, self.np_random = self.internal_reset(self.np_random, options)

        if return_info:
            return self.state.obs, self.state.info
        else:
            return self.state.obs

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, jp.ndarray, jp.ndarray, jp.ndarray, Dict[str, Any]]:
        self.state, self.np_random = self.internal_step(
            self.state, action, self.np_random
        )

        return (
            self.state.obs,
            self.state.reward,
            self.state.terminate,
            self.state.truncate,
            self.state.info,
        )


class VectorizeJaxEnv(gym.Env):
    def __init__(
        self,
        env: JaxEnv,
        n_envs: int,
        reset_device_parallelism: bool = False,
        step_device_parallelism: bool = False,
    ):
        assert n_envs > 0
        self.n_envs = n_envs

        if reset_device_parallelism:
            self.vectorise_reset = jax.pmap(
                env.internal_reset,
                in_axes=[0, None],
                static_broadcasted_argnums=1,
                axis_name="gym-reset",
                axis_size=n_envs,
            )
        else:
            jax.vmap(
                env.internal_reset,
                in_axes=[0, None],
                axis_name="gym-reset",
                axis_size=n_envs,
            )

        if step_device_parallelism:
            self.vectorise_step = jax.pmap(
                env.internal_step,
                in_axes=[0, 0, 0],
                axis_name="gym-step",
                axis_size=n_envs,
            )
        else:
            self.vectorise_step = jax.vmap(
                env.internal_step,
                in_axes=[0, 0, 0],
                axis_name="gym-step",
                axis_size=n_envs,
            )

        self.state: Optional[JaxState] = None
        _, seed = seeding.np_random()
        self.np_random: jp.ndarray = jax.random.split(jax.random.PRNGKey(seed), n_envs)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        if seed is not None:
            self.np_random = jax.random.split(jax.random.PRNGKey(seed), self.n_envs)
        self.state, self.np_random = self.vectorise_reset(self.np_random, options)

        if return_info:
            return self.state.obs
        else:
            return self.state.obs, self.state.info

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, jp.ndarray, jp.ndarray, jp.ndarray, Dict[str, Any]]:
        self.state, self.np_random = self.vectorise_step(
            self.state, action, self.np_random
        )

        return (
            self.state.obs,
            self.state.reward,
            self.state.terminate,
            self.state.truncate,
            self.state.info,
        )
