from typing import List

import pytest

import gym
from gym import envs
from gym.utils.env_checker import check_env
from tests.envs.spec_list import spec_list, spec_list_no_mujoco_py

# This runs a smoketest on each official registered env. We may want
# to try also running environments which are not officially registered
# envs.
IGNORE_WARNINGS = [
    "Agent's minimum observation space value is -infinity. This is probably too low.",
    "Agent's maximum observation space value is infinity. This is probably too high.",
    "We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html",
]
IGNORE_WARNINGS = [f"\x1b[33mWARN: {message}\x1b[0m" for message in IGNORE_WARNINGS]


@pytest.mark.parametrize("spec", spec_list, ids=[spec.id for spec in spec_list])
def test_env(spec):
    # Capture warnings
    env = spec.make(disable_env_checker=True)

    # Test if env adheres to Gym API
    with pytest.warns(None) as warnings:
        check_env(env)

    for warning in warnings.list:
        if warning.message.args[0] not in IGNORE_WARNINGS:
            raise gym.error.Error(f"Unexpected warning: {warning.message}")


@pytest.mark.parametrize(
    "spec", spec_list_no_mujoco_py, ids=[spec.id for spec in spec_list_no_mujoco_py]
)
def test_render_modes(spec):
    env = spec.make()

    for mode in env.metadata.get("render_modes", []):
        if mode != "human":
            new_env = spec.make(render_mode=mode)

            new_env.reset()
            new_env.step(new_env.action_space.sample())
            new_env.render()


def test_env_render_result_is_immutable():
    environs = [
        envs.make("Taxi-v3", render_mode="ansi"),
        envs.make("FrozenLake-v1", render_mode="ansi"),
    ]

    for env in environs:
        env.reset()
        output = env.render()
        assert isinstance(output, List)
        assert isinstance(output[0], str)
        env.close()
