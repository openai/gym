from gym.envs.safety.semi_supervised_pendulum_decay import SemiSupervisedPendulumDecayEnv
import numpy as np

env = SemiSupervisedPendulumDecayEnv()
for episode in xrange(100):
    env.reset()
    for i in xrange(100):
        print env.step(np.asarray([1]))
