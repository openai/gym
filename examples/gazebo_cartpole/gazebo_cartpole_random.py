import gym
import gym_gazebo
env = gym.make('GazeboCartPole-v0')
env.reset()
import time
# Check the env limits:
# print(env.action_space) # Box(3,)
# print(env.observation_space) # Box(9,)
import random
for i in range(100):
    env.reset()
    print("Reset!")
    sing = 1
    for x in range(200):
        # env.render()

        if( (x % 10) == 0):
            sing = sing*-1

        print("action = ", 1*sing, " ", x % 10)
        observation, reward, done, info = env.step(1*sing) # take a random action
        print("reward: ", reward, " observation: ", observation[0], " ", observation[2])
        if done: break
