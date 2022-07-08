from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import jumpy as jp
import numpy as np
from flax import struct

from gym import spaces
from gym.envs.jax_envs import JaxEnv, JaxState


@struct.dataclass
class CartpoleState:
    gravity = 9.8
    length = 0.5
    force_mag = 10.0
    tau = 0.02

    cart_mass = 1.0
    pole_mass = 0.1

    theta_threshold_radians = 12 * 2 * jnp.pi / 360
    x_threshold = 2.4

    total_mass = pole_mass + cart_mass
    pole_mass_length = pole_mass * length


def cartpole_reset(
    prng: jp.ndarray, options: Dict[str, Any]
) -> Tuple[JaxState[CartpoleState], jp.ndarray]:
    state = CartpoleState()

    prng, obs_rng = jax.random.split(prng)
    # state.obs = [x, x_dot, theta, theta_dot]
    obs = jax.random.uniform(obs_rng, (4,), minval=-0.05, maxval=0.05)

    return JaxState(state, obs), prng


def cartpole_step(
    state: JaxState[CartpoleState], action: jp.ndarray, prng: jp.ndarray
) -> Tuple[JaxState[CartpoleState], jp.ndarray]:
    # state.obs = [x, x_dot, theta, theta_dot]
    cos_theta, sin_theta = jnp.cos(state.obs[2]), jnp.sin(state.obs[2])
    force = jnp.where(action == 0, -state.state.force_mag, state.state.force_mag)

    # For the interested reader:
    # https://coneural.org/florian/papers/05_cart_pole.pdf
    temp = (
        force + state.state.pole_mass_length * jnp.square(state.obs[3]) * sin_theta
    ) / state.state.total_mass
    theta_acc = (sin_theta * state.state.gravity - cos_theta * temp) / (
        state.state.length
        * (
            4.0 / 3.0
            - state.state.pole_mass * jnp.square(cos_theta) / state.state.total_mass
        )
    )
    xacc = (
        temp
        - state.state.pole_mass_length * theta_acc * cos_theta / state.state.total_mass
    )

    x = state.obs[0] + state.state.tau * state.obs[1]
    x_dot = state.obs[1] + state.state.tau * xacc
    theta = state.obs[2] + state.state.tau * state.obs[3]
    theta_dot = state.obs[3] + state.state.tau * theta_acc

    obs = jnp.array([x, x_dot, theta, theta_dot], dtype=jnp.float32)
    terminated = jnp.any(
        jnp.array(
            [
                x < -state.state.x_threshold,
                x > state.state.x_threshold,
                theta < -state.state.theta_threshold_radians,
                theta > state.state.theta_threshold_radians,
            ]
        )
    )
    reward = jnp.int32(1.0)

    return state.replace(obs=obs, reward=reward, terminated=terminated), prng


class CartpoleEnv(JaxEnv[jnp.ndarray, int]):
    def __init__(self):
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        obs_high = jp.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        obs_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        action_space = spaces.Discrete(2)

        super().__init__(
            obs_space, action_space, cartpole_reset, cartpole_step, jit_fn=False
        )
