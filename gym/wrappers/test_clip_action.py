import numpy as np

import gym
from gym.wrappers import ClipAction


# mountaincar: action-based rewards
env = gym.make('MountainCarContinuous-v0')
wrapped_env = ClipAction(env)

seed = 0
env.seed(seed)
wrapped_env.seed(seed)

env.reset()
wrapped_env.reset()

actions = [[.4], [1.2], [-0.3], [0.0], [-2.5]]
for action in actions:
    _, r1, _, _ = env.step(np.clip(action, env.action_space.low, env.action_space.high))
    _, r2, _, _ = wrapped_env.step(action)
    assert np.allclose(r1, r2)
