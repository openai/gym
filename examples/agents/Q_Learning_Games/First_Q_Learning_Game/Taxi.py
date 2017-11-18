import gym
#import taxi env
from gym.envs.toy_text import taxi
#import tabular q agent
import tabular_q_agent_taxi

#libreria per generare grafici
import matplotlib.pyplot as plt
#lib to remove files
import os

env=taxi.TaxiEnv()#make TaxiEnv

agent=tabular_q_agent_taxi.TabularQAgent(env.observation_space,env.action_space)

print("BEGIN THE Q-LEARNING")
for i in range(50):
	agent.learn(env) #learn best choices and act on the env
	print("trial number: ",i)

print("Make the Rewards Plot")
        
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.figure(figsize=(10, 5))

f=open("rewards.txt","r")
stringa=f.readline()
n=0

while stringa!="":#count the number of rewards
	n+=1 
	stringa=f.readline()

newRewards=[ 0 for i in range(n)]

f=open("rewards.txt","r")
stringa=f.readline()
n=0

while stringa!="":#make the rewards list
	newRewards[n]=stringa
	n+=1
	stringa=f.readline()           

f.close()

#eps list with numRewards slots
eps=range(0,50)
        
plt.plot(eps,newRewards)
        
plt.title("Rewards collected over the time")
       
plt.xlabel("Trials")
plt.ylabel("Rewards")
plt.grid()#put the grid
 
plt.show()#print in output the plot and give the possibility to save it on your computer 

os.remove("/home/giacomo/Scrivania/Q_Learning_Games/First_Q_Learning_Game/rewards.txt")#to remove the file
