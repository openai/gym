"""
semi_supervised_pendulum_random is the pendulum task but where the reward function
is only given to the agent 1/10th of the time.

This is a toy problem but the principle is useful -- RL agents in the real world
will likely be learning from an inconsistent signal. For example, a human might
use a clicker to reward an RL agent but likely wouldn't do so with perfect consistency.

Note: In all semi_supervised environmenvts, we judge the RL agent based on their total
true_reward, not their percieved_reward. This means that even if the true_reward happens to
not be shown to the agent for an entire episode, the agent is still being judged
and should still perform as well as possible.
"""

from gym.envs.classic_control.pendulum import PendulumEnv

import numpy as np
import random

PROB_GET_REWARD = 0.1

class SemiSupervisedPendulumRandomEnv(PendulumEnv):
    def _step(self, action):
        observation, true_reward, done, info = super(SemiSupervisedPendulumRandomEnv, self)._step(action)

        if random.random() < PROB_GET_REWARD:
            perceived_reward = true_reward
        else:
            perceived_reward = 0

        return observation, perceived_reward, done, info
