#comparison between Sarsa q-learning and tabular-q-learning with matplotlib
import gym
#import roulette env
from gym.envs.toy_text import roulette
#import del tabular q agent_roulette
import tabular_q_agent_roulette_sarsa
import tabular_q_agent_roulette

wheel=roulette.RouletteEnv()#make the env

agentSarsa=tabular_q_agent_roulette_sarsa.TabularQAgentRoulette(wheel.observation_space,wheel.action_space)
 
agent=tabular_q_agent_roulette.TabularQAgentRoulette(wheel.observation_space,wheel.action_space)

print("BEGIN TABULAR Q-LEARNING")#sarsa Algo
for i in range(20):
	agent.learn(wheel) #learn best choices and act on the env
	print("trial number: %d" %i)

print("BEGIN SARSA")#tabular q-learning Algo
for i in range(20):
	agentSarsa.learn(wheel)
	print("trial number: %d" %i)

