#comparison between Sarsa q-learning and tabular-q-learning with matplotlib
import gym
#import taxi env
from gym.envs.toy_text import taxi
#import tabular q agent
import tabular_q_agent_taxi_sarsa
import tabular_q_agent_taxi

env=taxi.TaxiEnv()#make TaxiEnv

agentSarsa=tabular_q_agent_taxi_sarsa.TabularQAgentTaxi(env.observation_space,env.action_space)

print("BEGIN SARSA")#sarsa Algo
for i in range(1000):
	agentSarsa.learn(env) #learn best choices and act on the env
	print("trial number: %d" %i)
	
agent=tabular_q_agent_taxi.TabularQAgentTaxi(env.observation_space,env.action_space)

print("BEGIN TABULAR Q-LEARNING")#tabular q learning Algo
for i in range(1000):
	agent.learn(env)
	print("trial number: %d" %i)

