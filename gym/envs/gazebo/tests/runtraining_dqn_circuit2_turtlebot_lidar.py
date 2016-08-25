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
import json

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        print file
        os.unlink(file)

if __name__ == '__main__':

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    env = gym.make('GazeboCircuit2TurtlebotLidarNn-v0')
    outdir = '/tmp/gazebo_gym_experiments/'

    weights_path = '/tmp/turtle_c2_dqn_ep200.h5' 
    monitor_path = '/tmp/turtle_c2_dqn_ep200'
    params_json  = '/tmp/turtle_c2_dqn_ep200.json'

    #Load weights, monitor info and parameter info.
    #ADD TRY CATCH for this else
    with open(params_json) as outfile:
        d = json.load(outfile)
        epochs = d.get('epochs')
        steps = d.get('steps')
        updateTargetNetwork = d.get('updateTargetNetwork')
        explorationRate = d.get('explorationRate')
        minibatch_size = d.get('minibatch_size')
        learnStart = d.get('learnStart')
        learningRate = d.get('learningRate')
        discountFactor = d.get('discountFactor')
        memorySize = d.get('memorySize')
        network_inputs = d.get('network_inputs')
        network_outputs = d.get('network_outputs')
        network_layers = d.get('network_structure')
        current_epoch = d.get('current_epoch')

    deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
    deepQ.initNetworks(network_layers)
    deepQ.loadWeights(weights_path)

    observation = env.reset()
    while True:
        qValues = deepQ.getQValues(observation)
        action = deepQ.selectAction(qValues, 0)
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
    env.close()