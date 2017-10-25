#!/usr/bin/env python
import gym
import gym_gazebo
import time
import numpy
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import sarsa


if __name__ == '__main__':

    env = gym.make('GazeboCircuit2TurtlebotLidar-v0')

    outdir = '/tmp/gazebo_gym_experiments'
    env.monitor.start(outdir, force=True, seed=None)
    #plotter = LivePlot(outdir)

    last_time_steps = numpy.ndarray(0)

    sarsa = sarsa.Sarsa(actions=range(env.action_space.n),
                    epsilon=0.9, alpha=0.2, gamma=0.9)

    initial_epsilon = sarsa.epsilon

    epsilon_discount = 0.9986

    start_time = time.time()
    total_episodes = 10000
    highest_reward = 0

    for x in range(total_episodes):
        done = False

        cumulated_reward = 0 #Should going forward give more reward then L/R ?

        observation = env.reset()

        if sarsa.epsilon > 0.05:
            sarsa.epsilon *= epsilon_discount

        #render() #defined above, not env.render()

        state = ''.join(map(str, observation))

        for i in range(1500):

            # Pick an action based on the current state
            action = sarsa.chooseAction(state)

            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))
            nextAction = sarsa.chooseAction(nextState)
       
            #sarsa.learn(state, action, reward, nextState)
            sarsa.learn(state, action, reward, nextState, nextAction)

            env.monitor.flush(force=True)

            if not(done):
                state = nextState
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break 

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("EP: "+str(x+1)+" - [alpha: "+str(round(sarsa.alpha,2))+" - gamma: "+str(round(sarsa.gamma,2))+" - epsilon: "+str(round(sarsa.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))

    #Github table content
    print ("\n|"+str(total_episodes)+"|"+str(sarsa.alpha)+"|"+str(sarsa.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |")

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.monitor.close()
    env.close()
