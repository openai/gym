import gym
import gym_gazebo
env = gym.make('MARASide3DOF-v0')
env.reset()
import time
# Check the env limits:
# print(env.action_space) # Box(3,)
# print(env.observation_space) # Box(9,)
import random

for i in range(100):
    env.reset()
    time.sleep(1)
    # print("Reset!")
    for _ in range(50):
        observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
        time.sleep(0.1)
        # print("reward: ", reward)
