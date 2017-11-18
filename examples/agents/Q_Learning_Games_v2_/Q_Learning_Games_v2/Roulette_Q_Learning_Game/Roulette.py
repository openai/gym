import gym
#import roulette env
from gym.envs.toy_text import roulette
#import del tabular q agent_roulette
import tabular_q_agent_roulette

wheel=roulette.RouletteEnv()#make the env

agent=tabular_q_agent_roulette.TabularQAgentRoulette(wheel.observation_space,wheel.action_space)

print("BEGIN THE Q-LEARNING")
for i in range(100):
	agent.learn(wheel) #learn best choices and act on the env
	print("trial number: %d" %i)

agent.LogUpdate()

