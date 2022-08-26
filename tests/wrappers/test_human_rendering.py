import re

import pytest

import gym
from gym.wrappers import HumanRendering


def test_human_rendering():
    for mode in ["rgb_array", "single_rgb_array"]:
        env = HumanRendering(
            gym.make("CartPole-v1", render_mode=mode, disable_env_checker=True)
        )
        assert env.render_mode == "human"
        env.reset()

        for _ in range(75):
            _, _, done, _ = env.step(env.action_space.sample())
            if done:
                env.reset()

        env.close()

    env = gym.make("CartPole-v1", render_mode="human")
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Expected env.render_mode to be one of 'rgb_array' or 'single_rgb_array' but got 'human'"
        ),
    ):
        HumanRendering(env)
    env.close()
