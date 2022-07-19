"""
Implementation of a Jax-accelerated pendulum environment.
"""

from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey

from gym.functional import FuncEnv, StepReturn


class Pendulum(FuncEnv[jnp.ndarray, jnp.ndarray, int]):
    """Pendulum but in jax and functional."""

    max_speed = 8
    max_torque = 2.0
    dt = 0.05
    g = 10.0
    m = 1.0
    l = 1.0
    high_x = jnp.pi
    high_y = 1.0

    def initial(self, rng: PRNGKey):
        """Initial state generation."""
        high = jnp.array([self.high_x, self.high_y])
        return jax.random.uniform(key=rng, minval=-high, maxval=-high)

    def step(
        self, state: jnp.ndarray, action: Union[int, jnp.ndarray], rng: None = None
    ) -> StepReturn:
        """Pendulum transition."""
        th, thdot = state  # th := theta
        u = action

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        th_normalized = ((th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        costs = th_normalized**2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = jnp.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        new_state = jnp.array([newth, newthdot])
        return new_state, self.observation(new_state), -costs, False

    def observation(self, state: jnp.ndarray) -> jnp.ndarray:
        theta, thetadot = state
        return jnp.array([jnp.cos(theta), jnp.sin(theta), thetadot])
