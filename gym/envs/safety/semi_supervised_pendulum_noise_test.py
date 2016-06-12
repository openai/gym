from gym.envs.safety.semi_supervised_pendulum_noise import SemiSupervisedPendulumNoiseEnv
import numpy as np

env = SemiSupervisedPendulumNoiseEnv()
env.reset()
for i in xrange(100):
    print env.step(np.asarray([1]))
