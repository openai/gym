import gym
#import taxi env
from gym.envs.toy_text import taxi
#import tabular q agent
import tabular_q_agent_taxi

env=taxi.TaxiEnv()#make TaxiEnv

agent=tabular_q_agent_taxi.TabularQAgentTaxi(env.observation_space,env.action_space)

print("BEGIN THE Q-LEARNING")
for i in range(100):
	agent.learn(env) #learn best choices and act on the env
	print("trial number: %d" %i)

agent.LogUpdate()

