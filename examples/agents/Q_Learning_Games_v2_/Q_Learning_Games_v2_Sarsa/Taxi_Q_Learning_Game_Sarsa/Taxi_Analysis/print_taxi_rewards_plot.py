#libreria per generare grafici
import matplotlib.pyplot as plt
#lib to remove files
import os

print("Make the Rewards Plot")
        
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.figure(figsize=(10, 5))

f=open("rewards_taxi_sarsa.txt","r")
stringa=f.readline()
n=0

while stringa!="":#count the number of rewards
	n+=1 
	stringa=f.readline()

newRewards=[ 0 for i in range(n)]

f=open("rewards_taxi_sarsa.txt","r")
stringa=f.readline()
n=0

while stringa!="":#make the rewards list
	newRewards[n]=stringa
	n+=1
	stringa=f.readline()           

f.close()

#eps list with numRewards slots
eps=range(0,100)
        
plt.plot(eps,newRewards)
        
plt.title("Rewards collected over the time for Taxi game with Sarsa Algorithm")
       
plt.xlabel("Trials")
plt.ylabel("Rewards")
plt.grid()#put the grid
 
plt.show()#print in output the plot and give the possibility to save it on your computer 

os.remove("/home/giacomo/Scrivania/Q_Learning_Games_v2_/Q_Learning_Games_v2_Sarsa/Taxi_Q_Learning_Game_Sarsa/Taxi_Analysis/rewards_taxi_sarsa.txt")#to remove the file
