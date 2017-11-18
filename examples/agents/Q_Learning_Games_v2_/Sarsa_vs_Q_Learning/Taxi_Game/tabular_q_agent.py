#-*- coding: iso-8859-15 -*-
import sys
import numpy as np 
import collections
from collections import Iterable
from gym import spaces, utils
from gym.envs.toy_text import discrete

class TabularQAgent(object):

	def __init__(self, observation_space, action_space,**userconfig):#only with discrete env
    	
		if not isinstance(observation_space, spaces.Discrete):
			print('Observation space incompatible. (Only supports Discrete observation spaces.)')
			sys.exit(1)
		if not isinstance(action_space, spaces.Discrete):
			print('Action space incompatible. (Only supports Discrete action spaces.)')
			sys.exit(1)
    
		self.observation_space = observation_space
		self.action_space = action_space
		self.action_n = action_space.n
		self.config = {
			"init_mean" : 0.0,      # Initialize Q values with this mean
			"init_std" : 0.0,       # Initialize Q values with this standard deviation
			"learning_rate" : 0.1,
			"eps": 0.05,            # Epsilon in epsilon greedy policies
			"discount": 0.99,#from 0.95 to 0.99
			"n_iter": 10000}        # Number of iterations
		self.config.update(userconfig)
		self.q = collections.defaultdict(lambda: self.config["init_std"] * np.random.randn(self.action_n) + self.config["init_mean"])
		
	def act(self, observation, eps=None):
		if eps is None:
			eps = self.config["eps"]
        # epsilon greedy.
		if np.random.random() > 1-eps:#soft greedy       	
			if isinstance(self.q[observation],Iterable):
				action = np.argmax(self.q[observation])
		else:
			action=self.action_space.sample()
        	
		return action
	
	def learn(self, env):
		config = self.config
		obs = env.reset()
		q = self.q
		for t in range(config["n_iter"]):
			action = self.act(obs)
			obs2, reward, done, _ = env.step(action)
			future = 0.0
           	        
			if not done:
				future = np.max(q[obs2.item()])
			q[obs.item()][action] -= \
			    self.config["learning_rate"] * (q[obs.item()][action] - reward - config["discount"] * future)

			obs = obs2
