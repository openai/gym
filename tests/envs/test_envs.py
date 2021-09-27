import pytest
import numpy as np

from gym import envs
from tests.envs.spec_list import spec_list
from gym.spaces import Box
from gym.utils.env_checker import check_env


# This runs a smoketest on each official registered env. We may want
# to try also running environments which are not officially registered
# envs.
@pytest.mark.parametrize("spec", spec_list)
def test_env(spec):
    # Capture warnings
    with pytest.warns(None) as warnings:
        env = spec.make()

    # Test if env adheres to Gym API
    check_env(env, warn=True, skip_render_check=True)

    # Check that dtype is explicitly declared for gym.Box spaces
    for warning_msg in warnings:
        assert "autodetected dtype" not in str(warning_msg.message)

    ob_space = env.observation_space
    act_space = env.action_space
    ob = env.reset()
    assert ob_space.contains(ob), "Reset observation: {!r} not in space".format(ob)
    if isinstance(ob_space, Box):
        # Only checking dtypes for Box spaces to avoid iterating through tuple entries
        assert (
            ob.dtype == ob_space.dtype
        ), "Reset observation dtype: {}, expected: {}".format(ob.dtype, ob_space.dtype)

    a = act_space.sample()
    observation, reward, done, _info = env.step(a)
    assert ob_space.contains(observation), "Step observation: {!r} not in space".format(
        observation
    )
    assert np.isscalar(reward), "{} is not a scalar for {}".format(reward, env)
    assert isinstance(done, bool), "Expected {} to be a boolean".format(done)
    if isinstance(ob_space, Box):
        assert (
            observation.dtype == ob_space.dtype
        ), "Step observation dtype: {}, expected: {}".format(ob.dtype, ob_space.dtype)

    for mode in env.metadata.get("render.modes", []):
        env.render(mode=mode)

    # Make sure we can render the environment after close.
    for mode in env.metadata.get("render.modes", []):
        env.render(mode=mode)

    env.close()


# Run a longer rollout on some environments
def test_random_rollout():
    for env in [envs.make("CartPole-v0"), envs.make("FrozenLake-v1")]:
        agent = lambda ob: env.action_space.sample()
        ob = env.reset()
        for _ in range(10):
            assert env.observation_space.contains(ob)
            a = agent(ob)
            assert env.action_space.contains(a)
            (ob, _reward, done, _info) = env.step(a)
            if done:
                break
        env.close()


def test_env_render_result_is_immutable():
    environs = [
        envs.make("Taxi-v3"),
        envs.make("FrozenLake-v1"),
    ]

    for env in environs:
        env.reset()
        output = env.render(mode="ansi")
        assert isinstance(output, str)
        env.close()
