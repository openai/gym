"""Gotta go fast!  This trivial Env is meant for unit testing."""

import brax
import jax.numpy as jnp

from gym.envs.jax_env import JaxEnv, JaxState


class Fast(JaxEnv):
    """Trains an agent to go fast."""

    def __init__(self, **kwargs):
        super().__init__(config="dt: .02", **kwargs)

    def internal_reset(self, rng: jnp.ndarray) -> JaxState:
        zero = jnp.zeros(1)
        qp = brax.QP(pos=zero, vel=zero, rot=zero, ang=zero)
        obs = jnp.zeros(2)
        reward, terminate = jnp.zeros(2)
        return JaxState(qp, obs, reward, terminate)

    def internal_step(self, state: JaxState, action: jnp.ndarray) -> JaxState:
        vel = state.qp.vel + (action > 0) * self.sys.config.dt
        pos = state.qp.pos + vel * self.sys.config.dt

        qp = state.qp.replace(pos=pos, vel=vel)
        obs = jnp.array([pos[0], vel[0]])
        reward = pos[0]

        return state.replace(qp=qp, obs=obs, reward=reward)

    @property
    def observation_size(self):
        return 2

    @property
    def action_size(self):
        return 1
