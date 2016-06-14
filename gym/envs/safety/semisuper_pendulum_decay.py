"""
semisuper_pendulum_decay is the pendulum task but where the reward function
is given to the agent less and less often over time.

This is a toy problem but the principle is useful -- RL agents in the real world
will likely be learning from an inconsistent and decaying signal. For example, a human might
use a clicker to reward a household robot but might do so with less frequency over time.

Note: In all semisupervised environmenvts, we judge the RL agent based on their total
true_reward, not their percieved_reward. This means that even if the true_reward happens to
not be shown to the agent for an entire episode, the agent is still being judged
and should still perform as well as possible.
"""

from gym.envs.classic_control.pendulum import PendulumEnv

DECAY_RATE = 0.999

class SemisuperPendulumDecayEnv(PendulumEnv):
    def __init__(self):
        super(SemisuperPendulumDecayEnv, self).__init__()

        # This probability is only reset when you create a new instance of this env:
        self.prob_get_reward = 1.0

    def _step(self, action):
        observation, true_reward, done, info = super(SemisuperPendulumDecayEnv, self)._step(action)

        if self.np_random.uniform() < self.prob_get_reward:
            perceived_reward = true_reward
        else:
            perceived_reward = 0

        self.prob_get_reward *= DECAY_RATE

        return observation, perceived_reward, done, info
