'''
Created on May 14, 2016

@author: iaroslav
'''
import gym
import time

env = gym.make('ConvergenceControl-v0')
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