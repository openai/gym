#!/usr/bin/env python

'''
Based on: 
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''

import gym
import deepq
import time
from distutils.dir_util import copy_tree
import os

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym' + '.')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        os.unlink(file)

if __name__ == '__main__':

    env = gym.make('GazeboCircuit2TurtlebotLidarNn-v0')
    outdir = '/tmp/gazebo_gym_experiments'

    #Parameters needed to continue a training
    continue_execution = False
    #fill this if continue_execution=True
    weights_path = '/tmp/turtle_c2_dqn_ep100.h5' 
    monitor_path = '/tmp/turtle_c2_dqn_ep100'

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
    deepQ.initNetworks([100,100])

    if continue_execution:
        clear_monitor_files(outdir)
        copy_tree(monitor_path,outdir)
        env.monitor.start(monitor_path, resume=True, seed=None)
        deepQ.loadModel(weights_path)
    else:
        env.monitor.start(outdir, force=True, seed=None)

    stepCounter = 0
    highest_reward = 0

    start_time = time.time()

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
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    print ("\nEP "+str(epoch)+" - {} timesteps".format(t+1)+" - last100 Steps : "+str((sum(last100Scores)/len(last100Scores)))+" - Cumulated R: "+str(cumulated_reward)+"   Eps="+str(round(explorationRate, 2))+"     Time: %d:%02d:%02d" % (h, m, s))
                    if epoch%100==0:
                        #save model weights and monitoring data every 100 epochs.
                        deepQ.saveModel('/tmp/turtle_c2_dqn_ep'+str(epoch)+'.h5')
                        env.monitor.flush()
                        copy_tree(outdir,'/tmp/turtle_c2_dqn_ep'+str(epoch))
                break

            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                deepQ.updateTargetNetwork()
                print ("updating target network")

        explorationRate *= 0.995 #epsilon decay
        # explorationRate -= (2.0/epochs)
        explorationRate = max (0.05, explorationRate)

    env.monitor.close()
    env.close()