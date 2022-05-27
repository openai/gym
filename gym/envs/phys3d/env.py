"""A brax environment for training and inference."""

import abc
from typing import Any, Dict, Optional, Tuple, Union

import brax
import jax
import jumpy as jp
from brax.io import image
from flax import struct
from google.protobuf import text_format

import gym
from gym import spaces
from gym.core import ObsType


@struct.dataclass
class BraxState:
    """Environment state for training and inference."""

    qp: brax.QP
    obs: jp.ndarray
    reward: jp.ndarray
    terminate: jp.ndarray
    metrics: Dict[str, jp.ndarray] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


class BraxEnv(gym.Env, abc.ABC):
    """API for driving a brax system for training and inference."""

    def __init__(self, config: Optional[str], backend: Optional[str] = None):
        if config:
            config = text_format.Parse(config, brax.Config())
            self.sys = brax.System(config)

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 1 / self.sys.config.dt,
        }
        self.seed(0)
        self.backend = backend
        self.state = None

        def internal_reset(key):
            key1, key2 = jp.random_split(key)
            state = self.brax_reset(key2)
            return state, state.obs, key1

        self.internal_reset = jax.jit(internal_reset, backend=self.backend)

        def internal_step(state, action):
            state = self.brax_step(state, action)
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self.internal_step = jax.jit(internal_step, backend=self.backend)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        if seed is not None:
            self.np_random = jax.random.PRNGKey(seed)

        self.state, obs, self.np_random = self.internal_reset(self.np_random)
        # We return device arrays for pytorch users.
        return (obs, {}) if return_info else obs

    @abc.abstractmethod
    def brax_reset(self, rng: jp.ndarray) -> BraxState:
        """Resets the environment to an initial state."""

    def step(self, action):
        self.state, obs, reward, terminate, info = self.internal_step(
            self.state, action
        )
        # We return device arrays for pytorch users.
        return obs, reward, terminate, False, info

    @abc.abstractmethod
    def brax_step(self, state: BraxState, action: jp.ndarray) -> BraxState:
        """Run one timestep of the environment's dynamics."""

    @property
    def observation_size(self) -> int:
        """The size of the observation vector returned in step and reset."""
        rng = jp.random_prngkey(0)
        reset_state = self.unwrapped.brax_reset(rng)
        return reset_state.obs.shape[-1]

    @property
    def action_size(self) -> int:
        """The size of the action vector expected by step."""
        return self.sys.num_joint_dof + self.sys.num_forces_dof

    @property
    def unwrapped(self) -> "BraxEnv":
        return self

    def seed(self, seed: int = 0):
        self.np_random = jax.random.PRNGKey(seed)

    def render(self, mode: str = "rgb_array"):
        if mode == "rgb_array":
            sys, qp = self.sys, self.state.qp
            return image.render_array(sys, qp, 256, 256)
        else:
            return super().render(mode=mode)  # just raise an exception

    @property
    def observation_space(self):
        if hasattr(self, "_observation_space"):
            return self._observation_space
        else:
            obs_high = jp.inf * jp.ones(self.observation_size, dtype="float32")
            self._observation_space = spaces.Box(-obs_high, obs_high, dtype=jp.float32)
            return self._observation_space

    @property
    def action_space(self):
        if hasattr(self, "_action_space"):
            return self._action_space
        else:
            action_high = jp.ones(self.action_size, dtype="float32")
            self._action_space = spaces.Box(-action_high, action_high, dtype=jp.float32)
            return self._action_space
