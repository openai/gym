import gym

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
goal_steps = 200
score_requirement = -198
intial_games = 10000
#actions:  0 = LEFT; 1 = REST; 2=RIGHT
#observations: (position, velocity)

for step_index in range(1000):
    env.render()

    action = 2 # go to the right
    # Execute the action
    observation, reward, done, info = env.step(action)
    print("Step {}:".format(step_index))
    print("action: {}".format(action))
    print("observation: {}".format(observation))
    print("reward: {}".format(reward))
    print("done: {}".format(done))
    print("info: {}".format(info))

    print(observation)


env.close()
