#-*- coding: iso-8859-15 -*-
import sys
import os

import numpy as np 
import collections
from collections import Iterable
from gym import spaces, utils
from gym.envs.toy_text import discrete
import tabular_q_agent

#lib for file log
import time
import datetime

class TabularQAgentRoulette(tabular_q_agent.TabularQAgent):

    """
    Agent implementing tabular Q-learning for Roulette game.
    """

    def __init__(self, observation_space, action_space,**userconfig):#only with discrete envs
        super(TabularQAgentRoulette,self).__init__(observation_space, action_space,**userconfig)

    def softmax(self,q_value, beta=1.0):
        q_tilde = q_value - np.max(q_value)
        e_x = np.exp(beta * q_tilde)
        return e_x / np.sum(e_x)

    def select_a_with_softmax(self,obs, q_value, beta=1.0):
        prob_action = self.softmax(q_value[obs], beta=beta)
        cumsum_action = np.cumsum(prob_action)
        return np.where(np.random.rand() < cumsum_action)[0][0]

    def act(self, observation, eps=None):
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
        if np.random.random() > 1-eps:#soft greedy
            if isinstance(self.q[observation],Iterable):
                action = np.argmax(self.q[observation])
        else:
            action = self.select_a_with_softmax(observation,self.q)#no random action
        	               
        return action

    def learn(self, env):
        config = self.config
        obs = env.reset()
        q = self.q    
        action = self.act(obs)
        
        for t in range(config["n_iter"]):
        	obs2, reward, done, _ = env.step(action)
        	future=0.0
        	action2 = self.act(obs)
        	
        	if not done:
        		future = np.max(q[obs2])
        	q[obs][action] -= \
        		self.config["learning_rate"] * (q[obs][action] - reward - config["discount"] * future)
        		
        	obs = obs2
           	action=action2
        
        #scrivo su file le rewards..
        dirPath="Roulette_Analysis"#you can change the name path
        try:
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)#make folder in the current directory
        except OSError:
            print ('Error: Creating directory. ' +  dirPath)
        #scrivo su file le rewards..
        
        
        f=open("Roulette_Analysis/rewards_roulette_sarsa.txt","a")#a to append rewards mean!!
        
        nRewards=0
        sumRewards=0
        
        for i in q.keys():
            for j in q[i]:
                sumRewards+=j
                nRewards+=1
                 
        meanRewards=sumRewards/nRewards    
        f.write(str(meanRewards)+"\n")
        f.close()
        
    def LogUpdate(self):
       i = datetime.datetime.now()
       f=open("Roulette_Analysis/log.txt","a")
       f.write("INFO TRIAL:%s/%s/%s" %(i.day, i.month, i.year)+"---")
       f.write(time.strftime("%H:%M:%S")+"\n\n")
            
       f.write("\nuser config=%s\n" %self.config)
       
       f.write("observation space = %s\n"%self.observation_space)
       f.write("action space = %s\n"%self.action_space)
       f.write("LIST OF OBSERVATIONS:"+"\n")
       f.write("%s" %self.q.keys())
            
       f.write("\nQ-LEARNING TABLE WITH SARSA ALGORITHM (OBS AND ACTIONS REWARDS):"+"\n")
       f.write("SIZE OF THE TABLE:%s\n" %len(self.q))
       f.write("%s" %self.q.items())
       
       rewardsFile=open("Roulette_Analysis/rewards_roulette_sarsa.txt","r")
       f.write("\nLIST OF REWARDS MEAN FOR EACH EPISODE OF THIS TRIAL:\n%s\n" %rewardsFile.read())
       
       f.write("\n\n")
       f.close()
