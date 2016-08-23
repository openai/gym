#!/usr/bin/env python

'''
Based on: https://github.com/vmayoral/basic_reinforcement_learning
'''

import gym
import deepq

if __name__ == '__main__':

    env = gym.make('GazeboCircuit2TurtlebotLidarNn-v0')

    outdir = '/tmp/gazebo_gym_experiments'
    env.monitor.start(outdir, force=True, seed=None)
    #Each time we take a sample and update our weights it is called a mini-batch. 
    #Each time we run through the entire dataset, it's called an epoch.
    epochs = 1000
    steps = 10000
    updateTargetNetwork = 10000
    explorationRate = 1 #epsilon
    minibatch_size = 128
    learnStart = 128
    learningRate = 0.00025
    discountFactor = 0.99
    memorySize = 1000000

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False

    # turtlebot_nn_setup.bash must be executed.
    deepQ = deepq.DeepQ(100, 21, memorySize, discountFactor, learningRate, learnStart)
    # deepQ.initNetworks([30,30,30])
    # deepQ.initNetworks([30,30])
    deepQ.initNetworks([300,300])

    stepCounter = 0
    highest_reward = 0

    # number of reruns
    for epoch in xrange(epochs):
        observation = env.reset()
        cumulated_reward = 0

        # number of timesteps
        for t in xrange(steps):
            # env.render()
            qValues = deepQ.getQValues(observation)

            action = deepQ.selectAction(qValues, explorationRate)

            newObservation, reward, done, info = env.step(action)

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            deepQ.addMemory(observation, action, reward, newObservation, done)

            if stepCounter >= learnStart:
                if stepCounter <= updateTargetNetwork:
                    deepQ.learnOnMiniBatch(minibatch_size, False)
                else :
                    deepQ.learnOnMiniBatch(minibatch_size, True)

            observation = newObservation

            if (t >= 3000):
                print ("reached the end! :D")
                done = True

            if done:
                last100Scores[last100ScoresIndex] = t
                last100ScoresIndex += 1
                if last100ScoresIndex >= 100:
                    last100Filled = True
                    last100ScoresIndex = 0
                if not last100Filled:
                    print ("EP "+str(epoch)+" - {} timesteps".format(t+1)+"   Exploration="+str(round(explorationRate, 2)))
                else :
                    print ("EP "+str(epoch)+" - {} timesteps".format(t+1)+" - last100 Steps : "+str((sum(last100Scores)/len(last100Scores)))+" - Cumulated Reward : "+str(cumulated_reward)+"   Exploration="+str(round(explorationRate, 2)))
                break

            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                deepQ.updateTargetNetwork()
                print ("updating target network")

        explorationRate *= 0.995 #epsilon decay
        # explorationRate -= (2.0/epochs)
        explorationRate = max (0.05, explorationRate)
