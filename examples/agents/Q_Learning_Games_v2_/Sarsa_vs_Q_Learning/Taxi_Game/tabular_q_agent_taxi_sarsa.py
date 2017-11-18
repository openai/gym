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

class TabularQAgentTaxi(tabular_q_agent.TabularQAgent):
    """
    Agent implementing tabular Q-learning for Taxi game.
    """

    def __init__(self, observation_space, action_space,**userconfig):#funziona solo se l'ambiente è discreto
        super(TabularQAgentTaxi,self).__init__(observation_space, action_space,**userconfig)

    def learn(self, env):#@override
        config = self.config
        obs = env.reset()
        q = self.q
        action = self.act(obs)#choose the first action
        
        for t in range(config["n_iter"]):
            obs2, reward, done, _ = env.step(action)
            action2 = self.act(obs)#choose another action 
            future = 0.0
            
            """
            if t==1:
                print("INITIAL SITUATION")
                env.render()#print the initial state
            """
            #in TaxiEnv the observation space is only a integer!
            if not done:
            	future = np.max(q[obs2])
            q[obs][action] -= \
            	self.config["learning_rate"] * (q[obs][action] - reward - config["discount"] * future)
            	
            obs = obs2
            action=action2#update the action
       	"""
       	print("FINAL SITUATION")
        env.render()#print the final state
        """
        
        #scrivo su file le rewards..
        dirPath="Taxi_Analysis"#you can change the name path
        try:
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)#make folder in the current directory
        except OSError:
            print ('Error: Creating directory. ' +  dirPath)
        	
        f=open("Taxi_Analysis/rewards_taxi_sarsa.txt","a")#a to append rewards mean!!
        
        nRewards=0
        sumRewards=0
        
        for i in q.keys():
            #print("ricompense riscontrate nello stato numero: ",i)
            for j in q[i]:
                sumRewards+=j
                nRewards+=1
                 
        meanRewards=sumRewards/nRewards    
        f.write(str(meanRewards)+"\n")
        f.close()
        
    def LogUpdate(self):
       i = datetime.datetime.now()
       f=open("Taxi_Analysis/log.txt","a")
       f.write("INFO TRIAL:%s/%s/%s" %(i.day, i.month, i.year)+"---")
       f.write(time.strftime("%H:%M:%S")+"\n\n")
            
       f.write("\nuser config=%s\n" %self.config)
       
       f.write("observation space = %s\n"%self.observation_space)
       f.write("action space = %s\n"%self.action_space)
       f.write("LIST OF OBSERVATIONS:"+"\n")
       f.write("%s" %self.q.keys())
            
       f.write("\nQ-LEARNING TABLE (OBS AND ACTIONS REWARDS):"+"\n")
       f.write("SIZE OF THE TABLE:%s\n" %len(self.q))
       f.write("%s" %self.q.items())
       
       rewardsFile=open("Taxi_Analysis/rewards_taxi_sarsa.txt","r")
       f.write("\nLIST OF REWARDS MEAN FOR EACH EPISODE OF THIS TRIAL:\n%s\n" %rewardsFile.read())
       
       f.write("\n\n")
       f.close()
       
       
