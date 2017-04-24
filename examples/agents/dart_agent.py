__author__ = 'yuwenhao'

import gym
import numpy as np
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        env = gym.make(sys.argv[1])
    else:
        env = gym.make('DartWalker3dSPD-v1')

    env.env.disableViewer = False

    env.reset()

    for i in range(1000):
        env.step(env.action_space.sample())
        env.render()

    env.render(close=True)