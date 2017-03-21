__author__ = 'yuwenhao'

import gym
import numpy as np

env = gym.make('DartCartPole-v0')

ob = env.reset()
for i in range(10):
    env.render()
    ob, rew, done, _ = env.step([0])
    print(np.any(abs(ob-ob[0])))
