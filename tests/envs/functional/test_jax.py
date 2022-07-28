import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jrng = pytest.importorskip("jax.random")

from gym.envs.phys2d.cartpole import CartPoleF  # noqa: E402
from gym.envs.phys2d.pendulum import PendulumF  # noqa: E402


@pytest.mark.parametrize("env_class", [CartPoleF, PendulumF])
def test_normal(env_class):
    env = env_class()
    rng = jrng.PRNGKey(0)

    state = env.initial(rng)
    env.action_space.seed(0)

    for t in range(10):
        obs = env.observation(state)
        action = env.action_space.sample()
        next_state = env.transition(state, action, None)
        reward = env.reward(state, action, next_state)
        terminal = env.terminal(next_state)

        assert next_state.shape == state.shape
        try:
            float(reward)
        except ValueError:
            pytest.fail("Reward is not castable to float")
        try:
            bool(terminal)
        except ValueError:
            pytest.fail("Terminal is not castable to bool")

        assert next_state.dtype == jnp.float32
        assert isinstance(obs, jnp.ndarray)
        assert obs.dtype == jnp.float32

        state = next_state


@pytest.mark.parametrize("env_class", [CartPoleF, PendulumF])
def test_jit(env_class):
    env = env_class()
    rng = jrng.PRNGKey(0)

    env.transform(jax.jit)
    state = env.initial(rng)
    env.action_space.seed(0)

    for t in range(10):
        obs = env.observation(state)
        action = env.action_space.sample()
        next_state = env.transition(state, action, None)
        reward = env.reward(state, action, next_state)
        terminal = env.terminal(next_state)

        assert next_state.shape == state.shape
        try:
            float(reward)
        except ValueError:
            pytest.fail("Reward is not castable to float")
        try:
            bool(terminal)
        except ValueError:
            pytest.fail("Terminal is not castable to bool")

        assert next_state.dtype == jnp.float32
        assert isinstance(obs, jnp.ndarray)
        assert obs.dtype == jnp.float32

        state = next_state


@pytest.mark.parametrize("env_class", [CartPoleF, PendulumF])
def test_vmap(env_class):
    env = env_class()
    num_envs = 10
    rng = jrng.split(jrng.PRNGKey(0), num_envs)

    env.transform(jax.vmap)
    env.transform(jax.jit)
    state = env.initial(rng)
    env.action_space.seed(0)

    for t in range(10):
        obs = env.observation(state)
        action = jnp.array([env.action_space.sample() for _ in range(num_envs)])
        # if isinstance(env.action_space, Discrete):
        #     action = action.reshape((num_envs, 1))
        next_state = env.transition(state, action, None)
        terminal = env.terminal(next_state)
        reward = env.reward(state, action, next_state)

        assert next_state.shape == state.shape
        assert next_state.dtype == jnp.float32
        assert reward.shape == (num_envs,)
        assert reward.dtype == jnp.float32
        assert terminal.shape == (num_envs,)
        assert terminal.dtype == np.bool
        assert isinstance(obs, jnp.ndarray)
        assert obs.dtype == jnp.float32

        state = next_state
