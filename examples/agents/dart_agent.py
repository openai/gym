__author__ = 'yuwenhao'

import gym
import numpy as np

if __name__ == '__main__':
    env = gym.make('DartWalker3d-v1')

    env.reset()

    for i in range(1000):
        env.step(env.action_space.sample())
        env.render()

    env.render(close=True)