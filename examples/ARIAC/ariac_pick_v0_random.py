import gym
import gym_gazebo
env = gym.make('ARIACPick-v0')
env.reset()
import time

import random

for i in range(100):
    env.reset()
    print("Reset!")
    for x in range(200):
        observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
        # if done: break
