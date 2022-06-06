import numpy as np
import pytest

from gym import envs
from gym.spaces import Box
from gym.utils.env_checker import check_env
from tests.envs.spec_list import spec_list, spec_list_no_mujoco_py


# This runs a smoketest on each official registered env. We may want
# to try also running environments which are not officially registered
# envs.
@pytest.mark.filterwarnings(
    "ignore:.*We recommend you to use a symmetric and normalized Box action space.*"
)
@pytest.mark.parametrize(
    "spec", spec_list_no_mujoco_py, ids=[spec.id for spec in spec_list_no_mujoco_py]
)
def test_env(spec):
    # Capture warnings
    with pytest.warns(None) as warnings:
        env = spec.make()

    # Test if env adheres to Gym API
    check_env(env, skip_render_check=True)

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
    for mode in env.metadata.get("render_modes", []):
        if not (mode == "human" and spec.entry_point.startswith("gym.envs.mujoco")):
            env.render(mode=mode)

    # Make sure we can render the environment after close.
    for mode in env.metadata.get("render_modes", []):
        if not (mode == "human" and spec.entry_point.startswith("gym.envs.mujoco")):

            env.render(mode=mode)

    env.close()


@pytest.mark.parametrize("spec", spec_list, ids=[spec.id for spec in spec_list])
def test_reset_info(spec):

    with pytest.warns(None):
        env = spec.make()

    ob_space = env.observation_space
    obs = env.reset()
    assert ob_space.contains(obs)
    obs = env.reset(return_info=False)
    assert ob_space.contains(obs)
    obs, info = env.reset(return_info=True)
    assert ob_space.contains(obs)
    assert isinstance(info, dict)
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
