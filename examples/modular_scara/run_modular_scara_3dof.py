import gym
import gym_gazebo
env = gym.make('GazeboModularScara3DOF-v0')
env.reset()

# Check the env limits:
# print(env.action_space) # Box(3,)
# print(env.observation_space) # Box(9,)

for _ in range(1000):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print("reward: ", reward)
