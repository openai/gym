import gym
import numpy as np
import pytest
from gym.spaces.box import Box
from gym.utils.env_checker import check_env


# Testing with different non-default kwargs,
@pytest.mark.parametrize("depth", [True, False])
@pytest.mark.parametrize("labels", [True, False])
@pytest.mark.parametrize("position", [True, False])
@pytest.mark.parametrize("health", [True, False])
def test_vizdoom_env(depth, labels, position, health):
    # Capture warnings
    with pytest.warns(None) as warnings:
        env = gym.make(
            "VizdoomTakeCover-v0",
            depth=depth,
            labels=labels,
            position=position,
            health=health,
        )

    # Test if env adheres to Gym API
    check_env(env, warn=True, skip_render_check=True)

    # Check that dtype is explicitly declared for gym.Box spaces
    for warning_msg in warnings:
        assert "autodetected dtype" not in str(warning_msg.message)

    ob_space = env.observation_space
    act_space = env.action_space
    ob = env.reset()
    assert ob_space.contains(ob), f"Reset observation: {ob!r} not in space"
    if isinstance(ob_space, Box):
        # Only checking dtypes for Box spaces to avoid iterating through tuple entries
        assert (
            ob.dtype == ob_space.dtype
        ), f"Reset observation dtype: {ob.dtype}, expected: {ob_space.dtype}"

    a = act_space.sample()
    observation, reward, done, _info = env.step(a)
    assert ob_space.contains(
        observation
    ), f"Step observation: {observation!r} not in space"
    assert np.isscalar(reward), f"{reward} is not a scalar for {env}"
    assert isinstance(done, bool), f"Expected {done} to be a boolean"

    env.close()


# Check obs on terminal state
@pytest.mark.parametrize("depth", [True, False])
@pytest.mark.parametrize("labels", [True, False])
@pytest.mark.parametrize("position", [True, False])
@pytest.mark.parametrize("health", [True, False])
def test_terminal_state(depth, labels, position, health):
    env = gym.make(
        "VizdoomTakeCover-v0",
        depth=depth,
        labels=labels,
        position=position,
        health=health,
    )

    agent = lambda ob: env.action_space.sample()
    ob = env.reset()
    done = False
    while not done:
        a = agent(ob)
        (ob, _reward, done, _info) = env.step(a)
        if done:
            break
        env.close()
    assert env.observation_space.contains(ob)
