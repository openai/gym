import gym
import gym_gazebo
env = gym.make('GazeboCartPole-v0')
env.reset()
import time
# Check the env limits:
# print(env.action_space) # Box(3,)
# print(env.observation_space) # Box(9,)

for i in range(10):
    env.reset()
    print("Reset!")
    for _ in range(200):
        # env.render()
        observation, reward, done, info = env.step(1) # take a random action
        print("reward: ", reward)
        time.sleep(1)
