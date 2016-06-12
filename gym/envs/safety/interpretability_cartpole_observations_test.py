from gym.envs.safety.interpretability_cartpole_observations import InterpretabilityCartpoleObservationsEnv

import numpy as np

env = InterpretabilityCartpoleObservationsEnv()
env.reset()
obs = np.asarray([0., 0., 0., 0.])
for i in xrange(100):
    obs, reward, done, info = env.step(
        [1,
        obs,
        obs,
        obs,
        obs,
        obs]
    )

    print reward
