"""
Implementation of a Jax-accelerated cartpole environment.
"""

from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey

from gym.functional import FuncEnv, StepReturn


class CartPole(FuncEnv[jnp.ndarray, jnp.ndarray, int]):
    """Cartpole but in jax and functional.

    Example usage:
    ```
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(0)

    env = CartPole({"x_init": 0.5})
    state = env.initial(key)
    print(state)
    print(env.step(state, 0))

    env.transform(jax.jit)

    state = env.initial(key)
    print(state)
    print(env.step(state, 0))

    vkey = jax.random.split(key, 10)
    env.transform(jax.vmap)
    vstate = env.initial(vkey)
    print(vstate)
    print(env.step(vstate, jnp.array([0 for _ in range(10)])))
    ```
    """

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5
    polemass_length = masspole + length
    force_mag = 10.0
    tau = 0.02
    theta_threshold_radians = 12 * 2 * np.pi / 360
    x_threshold = 2.4
    x_init = 0.05

    def initial(self, rng: PRNGKey):
        """Initial state generation."""
        return jax.random.uniform(
            key=rng, minval=-self.x_init, maxval=self.x_init, shape=(4,)
        )

    def step(
        self, state: jnp.ndarray, action: Union[int, jnp.ndarray], rng: None = None
    ) -> StepReturn:
        """Cartpole transition."""
        x, x_dot, theta, theta_dot = state
        force = jnp.sign(action - 0.5) * self.force_mag
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        terminated = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

        reward = jax.lax.cond(terminated, lambda: 0.0, lambda: 1.0)

        state = jnp.array((x, x_dot, theta, theta_dot), dtype=jnp.float32)

        return state, self.observation(state), reward, terminated

    def observation(self, state: jnp.ndarray) -> jnp.ndarray:
        """Cartpole observation."""
        return state
