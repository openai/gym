import gym
import gym_gazebo
env = gym.make('GazeboModularScara3DOF-v0')
env.reset()
for _ in range(1000):
    # env.render()
    env.step(env.action_space.sample()) # take a random action
