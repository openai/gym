import numpy as np

import gym
from gym.wrappers import ClipAction


def test_clip_action():
    # mountaincar: action-based rewards
    env = gym.make("MountainCarContinuous-v0", disable_env_checker=True)
    wrapped_env = ClipAction(
        gym.make("MountainCarContinuous-v0", disable_env_checker=True)
    )

    seed = 0

    env.reset(seed=seed)
    wrapped_env.reset(seed=seed)

    actions = [[0.4], [1.2], [-0.3], [0.0], [-2.5]]
    for action in actions:
        obs1, r1, ter1, trunc1, _ = env.step(
            np.clip(action, env.action_space.low, env.action_space.high)
        )
        obs2, r2, ter2, trunc2, _ = wrapped_env.step(action)
        assert np.allclose(r1, r2)
        assert np.allclose(obs1, obs2)
        assert ter1 == ter2
        assert trunc1 == trunc2
