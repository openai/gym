"""
semi_supervised_pendulum_noise is the pendulum task but where reward function is noisy.

This is a toy problem but the principle is useful -- RL agents in the real world
will likely be learning from a noisy signal. Either because their sensors are noisy or
because humans providing the reward signal are not doing a perfect job. Or both.

Note: In all semi_supervised environmenvts, we judge the RL agent based on their total
true_reward, not their percieved_reward. This means that even though the reward that the
agent sees is stochastic, the true reward by which they are judged is a (usually deterministic)
function of just the state of the environment and the agent's actions.
"""

from gym.envs.classic_control.pendulum import PendulumEnv

NOISE_STANDARD_DEVIATION = 3.0

class SemiSupervisedPendulumNoiseEnv(PendulumEnv):
    def _step(self, action):
        observation, true_reward, done, info = super(SemiSupervisedPendulumNoiseEnv, self)._step(action)

        perceived_reward = true_reward + self.np_random.normal(scale=NOISE_STANDARD_DEVIATION)

        return observation, perceived_reward, done, info
