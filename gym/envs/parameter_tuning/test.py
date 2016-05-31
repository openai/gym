import gym

env = gym.make('CNNClassifierTraining-v0')
for i_episode in xrange(20):

    observation = env.reset()

    for t in xrange(100):

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()

        if done:
            print "Final result:"
            env.render()
            break