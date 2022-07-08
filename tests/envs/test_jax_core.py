import numpy as np
import pytest

from gym.envs.classic_control.cartpole import CartPoleEnv as CartPoleNpEnv

pytest.importorskip("jax")

from gym.envs.phys2d.cartpole import CartpoleEnv as CartpoleJnpEnv  # noqa: E402


def test_phys2d_cc_cartpole():
    print()
    jax_env = CartpoleJnpEnv()
    np_env = CartPoleNpEnv()

    jax_obs = jax_env.reset()
    np_env.reset()

    np_env.state = np.array(jax_obs)
    assert jax_env.state is not None
    assert np.all(np_env.state == jax_env.state.obs)

    done = False
    i = 1
    while not done:
        action = jax_env.action_space.sample()
        jax_obs, jax_reward, jax_terminated, jax_truncated, jax_info = jax_env.step(
            action
        )
        np_obs, np_reward, np_terminated, np_info = np_env.step(action)
        done = jax_terminated or jax_truncated

        assert np.allclose(jax_obs, np_obs, atol=1.0e-5)
        assert np.allclose(jax_reward, np_reward)
        assert np.allclose(jax_terminated, np_terminated)
        assert jax_info == np_info
