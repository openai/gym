#libreria per generare grafici
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#lib to remove files
import os

print("Make the comparison Plots")
        
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.figure(figsize=(10, 5))

f=open("rewards_roulette_qlearning.txt","r")
stringa=f.readline()
n=0

while stringa!="":#count the number of rewards
	n+=1 
	stringa=f.readline()

newRewards=[ 0 for i in range(n)]
newRewardsSarsa=[ 0 for i in range(n)]

#read q-learning rewards
f=open("rewards_roulette_qlearning.txt","r")
stringa=f.readline()
n=0

while stringa!="":#make the rewards list
	newRewards[n]=stringa
	n+=1
	stringa=f.readline()           

f.close()

#read sarsa rewards
f=open("rewards_roulette_sarsa.txt","r")
stringa=f.readline()
n=0
while stringa!="":#make the rewards list
	newRewardsSarsa[n]=stringa
	n+=1
	stringa=f.readline()           

f.close()

#eps list with numRewards slots
eps=range(0,20)
        
plt.plot(eps,newRewards,'r',eps,newRewardsSarsa,'b')


plt.title("Rewards collected over the time for Roulette game")
       
plt.xlabel("Trials")
plt.ylabel("Rewards")
plt.grid()#put the grid

qlearningLegend = mpatches.Patch(color='red', label='Q-learning')
SarsaLegend = mpatches.Patch(color='blue', label='Sarsa')
plt.legend(handles=[qlearningLegend,SarsaLegend])

plt.show()#print in output the plot and give the possibility to save it on your computer 

os.remove("/home/giacomo/Scrivania/Q_Learning_Games_v2_/Sarsa_vs_Q_Learning/Roulette_Game/Roulette_Analysis/rewards_roulette_sarsa.txt")
os.remove("/home/giacomo/Scrivania/Q_Learning_Games_v2_/Sarsa_vs_Q_Learning/Roulette_Game/Roulette_Analysis/rewards_roulette_qlearning.txt")#to remove the file
