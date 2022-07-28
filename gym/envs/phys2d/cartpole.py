"""
Implementation of a Jax-accelerated cartpole environment.
"""

from typing import Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jax.random import PRNGKey
from pygame import gfxdraw

import gym
from gym.functional import ActType, FuncEnv, RenderStateType, StateType


class CartPoleF(FuncEnv[jnp.ndarray, jnp.ndarray, int, float, bool]):
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

    screen_width = 600
    screen_height = 400

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def initial(self, rng: PRNGKey):
        """Initial state generation."""
        return jax.random.uniform(
            key=rng, minval=-self.x_init, maxval=self.x_init, shape=(4,)
        )

    def transition(
        self, state: jnp.ndarray, action: Union[int, jnp.ndarray], rng: None = None
    ) -> StateType:
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

        state = jnp.array((x, x_dot, theta, theta_dot), dtype=jnp.float32)

        return state

    def observation(self, state: jnp.ndarray) -> jnp.ndarray:
        """Cartpole observation."""
        return state

    def terminal(self, state: jnp.ndarray) -> jnp.ndarray:
        x, _, theta, _ = state

        terminated = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

        return terminated

    def reward(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> jnp.ndarray:
        x, _, theta, _ = state

        terminated = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

        reward = jax.lax.cond(terminated, lambda: 0.0, lambda: 1.0)
        return reward

    def render_image(
        self, state: StateType, render_state: Tuple[pygame.Surface, pygame.time.Clock]
    ) -> Tuple[RenderStateType, np.ndarray]:

        screen, clock = render_state

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        x = state

        surf = pygame.Surface((self.screen_width, self.screen_height))
        surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(surf, 0, self.screen_width, carty, (0, 0, 0))

        surf = pygame.transform.flip(surf, False, True)
        screen.blit(surf, (0, 0))

        return (screen, clock), np.transpose(
            np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
        )
