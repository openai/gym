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
    if isinstance(ob_space, Box):
        assert (
            observation.dtype == ob_space.dtype
        ), f"Step observation dtype: {ob.dtype}, expected: {ob_space.dtype}"

    for mode in env.metadata.get("render.modes", []):
        env.render(mode=mode)

    # Make sure we can render the environment after close.
    for mode in env.metadata.get("render.modes", []):
        env.render(mode=mode)

    env.close()


def test_reset_info():
    env_names = ["CartPole-v0", "CartPole-v1","MountainCar-v0","MountainCarContinuous-v0","Pendulum-v1"
    ,"Acrobot-v1","LunarLander-v2","LunarLanderContinuous-v2","BipedalWalker-v3","BipedalWalkerHardcore-v3"
    ,"CarRacing-v0","Blackjack-v1","FrozenLake-v1","FrozenLake8x8-v1","CliffWalking-v0","Taxi-v3"
    ,"Reacher-v2","Pusher-v2","Thrower-v2","Striker-v2","InvertedPendulum-v2","InvertedDoublePendulum-v2"
    ,"HalfCheetah-v2","HalfCheetah-v3","Hopper-v2","Hopper-v3","Swimmer-v2","Swimmer-v3","Walker2d-v2"
    ,"Walker2d-v3","Ant-v2","Ant-v3","Humanoid-v2","Humanoid-v3","HumanoidStandup-v2"]
    for env_name in env_names:
        env = envs.make(env_name)
        obs = env.reset()
        assert (isinstance(obs, np.ndarray) or isinstance(obs,tuple) or isinstance(obs, int))
        del obs
        obs,info = env.reset(return_info = True)
        assert (isinstance(obs, np.ndarray) or isinstance(obs,tuple) or isinstance(obs, int))
        assert (isinstance(info, dict))
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
