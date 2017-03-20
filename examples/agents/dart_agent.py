__author__ = 'yuwenhao'

import gym

if __name__ == '__main__':
    env = gym.make('DartHopper-v1')

    env.reset()

    for i in range(1000):
        env.step([5,3,1])
        env.render()

    env.render(close=True)