"""Module docstring."""
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey

from gym import Space

StateType = TypeVar("StateType")
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")
StepReturn = Tuple[StateType, ObsType, float, bool, bool, Dict[str, Any]]


class FuncEnv(Generic[StateType, ObsType, ActType]):
    """Base functional env class."""

    observation_space: Space
    action_space: Space

    def initial(self, rng: Any = None) -> StateType:
        """Initial state."""
        raise NotImplementedError

    def step(self, state: StateType, action: ActType, rng: Any = None) -> StepReturn:
        """Transition."""
        raise NotImplementedError

    def transform(self, func: Callable[[Callable], Callable]):
        """Functional transformations."""
        self.initial = func(self.initial)
        self.step = func(self.step)


class JaxPole(FuncEnv[jnp.ndarray, jnp.ndarray, int]):
    """Cartpole but in jax and functional."""

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5
    polemass_length = masspole + length
    force_mag = 10.0
    tau = 0.02
    kinematics_integrator = "euler"
    theta_threshold_radians = 12 * 2 * np.pi / 360
    x_threshold = 2.4

    def initial(self, rng: Optional[PRNGKey] = None):
        """Initial state generation."""
        if rng is None:
            rng = jax.random.PRNGKey(np.random.default_rng().integers(0, 2**32 - 1))
        return jax.random.uniform(key=rng, minval=-0.05, maxval=0.05, shape=(4,))

    def step(self, state: StateType, action: ActType, rng: Optional[PRNGKey] = None):
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

        return state, state, reward, terminated, False, {}
