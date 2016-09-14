#!/usr/bin/env python

'''
Based on: 
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
import gym
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json
import random
import numpy as np
from keras.models import Sequential, load_model
from keras.initializations import normal
from keras import optimizers
from keras.optimizers import RMSprop
from keras.layers import Convolution2D, Flatten, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import SGD , Adam
import memory

class DeepQ:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s')

    """
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        """
        Parameters:
            - inputs: input size
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.input_size = inputs
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate

    def initNetworks(self, hiddenLayers):
        model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.model = model

        targetModel = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.targetModel = targetModel

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        '''model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4),input_shape=(1,80,80)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1,1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(3))
        adam = Adam(lr=1e-6)
        model.compile(loss='mse',optimizer=adam)
        model.add(Convolution2D(32, 3, 3, input_shape=(1,80,80)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(96))
        model.add(Activation('relu'))
        model.add(Dense(3))
        model.compile(RMSprop(), 'MSE')
        model.summary()'''

        #Flapply bird cnn
        model = Sequential()
        model.add(Convolution2D(16, 3, 3, subsample=(2,2), input_shape=(img_channels,img_rows,img_cols)))
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(16, 3, 3, subsample=(2,2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        #model.add(Dropout(0.5))

        #model.add(ZeroPadding2D((1, 1)))
        #model.add(Convolution2D(32, 3, 3))
        #model.add(Activation('relu'))
        #model.add(ZeroPadding2D((1, 1)))
        #model.add(Convolution2D(32, 3, 3))
        #model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        ##model.add(Dropout(0.5))
#
#        #model.add(ZeroPadding2D((1, 1)))
#        #model.add(Convolution2D(64, 3, 3))
#        #model.add(Activation('relu'))
#        #model.add(ZeroPadding2D((1, 1)))
#        #model.add(Convolution2D(64, 3, 3))
#        #model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))


        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(network_outputs))

        #adam = Adam(lr=1e-6)
        #model.compile(loss='mse',optimizer=adam)

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mse',optimizer=sgd)
        model.summary()

        return model

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print "layer ",i,": ",weights
            i += 1


    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    # predict Q values for all the actions
    def getQValues(self, state):
        predicted = self.model.predict(state)
        return predicted[0]

    def getTargetQValues(self, state):
        predicted = self.targetModel.predict(state)
        return predicted[0]

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else : 
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def selectActionByProbability(self, qValues, bias):
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if (rand <= value):
                return i
            i += 1

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        # Do not learn until we've got self.learnStart samples        
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            X_batch = np.empty((1,img_channels,img_rows,img_cols), dtype = np.float64)
            Y_batch = np.empty((1,self.output_size), dtype = np.float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else :
                    qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

                #print "Batch.shape %s, state.shape %s" % (str(X_batch.shape), str(state.shape))

                X_batch = np.append(X_batch, state.copy(), axis=0)

                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, newState.copy(), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
            #print("model FIT debug-------")
            #print(X_batch.shape)
            #print(Y_batch.shape)
            self.model.fit(X_batch, Y_batch, validation_split=0.2, batch_size = len(miniBatch), nb_epoch=1, verbose = 0)

    def saveModel(self, path):
        self.model.save(path)

    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())

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
    env = gym.make('GazeboCircuit2cTurtlebotCameraNnEnv-v0')
    outdir = '/tmp/gazebo_gym_experiments/'

    continue_execution = False
    #fill this if continue_execution=True

    weights_path = '/tmp/turtle_c2_dqn_ep1000.h5' 
    monitor_path = '/tmp/turtle_c2_dqn_ep1000'
    params_json  = '/tmp/turtle_c2_dqn_ep1000.json'

    img_rows, img_cols, img_channels = env.img_rows, env.img_cols, env.img_channels


    if not continue_execution:
        #Each time we take a sample and update our weights it is called a mini-batch. 
        #Each time we run through the entire dataset, it's called an epoch.
        #PARAMETER LIST
        epochs = 100000
        steps = 1000
        updateTargetNetwork = 10000000000#5000
        explorationRate = 1
        finalExplorationRate = 0.01
        minibatch_size = 32
        learnStart = 5000
        learningRate = 0.00025
        discountFactor = 0.95
        memorySize = 100000
        network_inputs = 100 #NOT USED
        network_outputs = 3
        network_structure = [300,300]
        explorationDecay = 0.995

        current_epoch = 0

        deepQ = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)
        env.monitor.start(outdir, force=True, seed=None)
    else:
        #Load weights, monitor info and parameter info.
        #ADD TRY CATCH fro this else
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
            img_rows = d.get('img_rows')
            img_cols = d.get('img_cols')
            img_channels = d.get('img_channels')

        deepQ = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_layers)
        deepQ.loadWeights(weights_path)

        clear_monitor_files(outdir)
        copy_tree(monitor_path,outdir)
        env.monitor.start(outdir, resume=True, seed=None)

    last100Rewards = [0] * 100
    last100RewardsIndex = 0
    last100Filled = False
    stepCounter = 0
    highest_reward = 0

    start_time = time.time()

    #start iterating from 'current epoch'.


    for epoch in xrange(current_epoch+1, epochs+1, 1):
        observation = env.reset()
        cumulated_reward = 0

        # number of timesteps
        for t in xrange(steps):
            # env.render()
            qValues = deepQ.getQValues(observation)

            # manual pre training for 2 EPS
            '''if epoch<3:
                #PRETRAINING
                print("Input action: ") #1,2,3 -> 0.1.2
                key = getch()
                action = int(key)-1
            else:
                action = deepQ.selectAction(qValues, explorationRate)'''

            action = deepQ.selectAction(qValues, explorationRate)
            newObservation, reward, done, info = env.step(action)

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            deepQ.addMemory(observation, action, reward, newObservation, done)


            #if learnStart == stepCounter:
            if explorationRate == learnStart:
                print("Starting learning")

            #if stepCounter >= learnStart:
            if explorationRate <= 0.2:
                if stepCounter <= updateTargetNetwork:
                    deepQ.learnOnMiniBatch(minibatch_size, False)
                else :
                    deepQ.learnOnMiniBatch(minibatch_size, True)

            observation = newObservation

            if (t >= 1000):
                print ("reached the end")
                done = True

            env.monitor.flush(force=True)

            if done:
                last100Rewards[last100RewardsIndex] = cumulated_reward
                last100RewardsIndex += 1
                if last100RewardsIndex >= 100:
                    last100Filled = True
                    last100RewardsIndex = 0
                if not last100Filled:
                    print ("EP "+str(epoch)+" - {} timesteps".format(t+1)+"   Exploration="+str(round(explorationRate, 2)))
                else :
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    print ("EP "+str(epoch)+" - {} timesteps".format(t+1)+" - last100 C_Rewards : "+str(int((sum(last100Rewards)/len(last100Rewards))))+" - Cumulated R: "+str(cumulated_reward)+"   Eps="+str(round(explorationRate, 2))+"     Time: %d:%02d:%02d" % (h, m, s))
                    if (epoch)%100==0:
                        #save model weights and monitoring data every 100 epochs. 
                        deepQ.saveModel('/tmp/turtle_c2_dqn_ep'+str(epoch)+'.h5')
                        env.monitor.flush()
                        copy_tree(outdir,'/tmp/turtle_c2_dqn_ep'+str(epoch))
                        #save simulation parameters.
                        parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch', 'img_rows', 'img_cols', 'img_channels']
                        parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch, img_rows, img_cols, img_channels]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open('/tmp/turtle_c2_dqn_ep'+str(epoch)+'.json', 'w') as outfile:
                            json.dump(parameter_dictionary, outfile)
                break

            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                deepQ.updateTargetNetwork()
                print ("updating target network. total steps: "+str(stepCounter))
            if stepCounter % 2500 == 0:
                print("stepCounter = "+str(stepCounter))

        explorationRate *= explorationDecay
        # explorationRate -= (2.0/epochs)
        explorationRate = max (finalExplorationRate, explorationRate)

    env.monitor.close()
    env.close()
