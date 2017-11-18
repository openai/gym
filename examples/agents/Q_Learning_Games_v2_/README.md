### Q_Agent_Implementation

An implementation of a Q-Agent and Sarsa Agent using OpenAI Gym toolkit as support.

### Prerequisites

1. 	You have to know Python programming language.
2. 	You have to install OpenAI Gym toolkit. 
	If you have not installed OpenAI Gym yet follow the instructions here: https://github.com/openai/gym#openai-gym. You can do a minimal or a full 	installation. Otherwise you can install envs you need by hand.

### After Installation: recommendations about how to write the code

After the installation of the OpenAI Gym you won't need to install anything else. 
We will use the file "tabular_q_agent.py" contained in examples/agents as starting point. I suggest you to copy this file because it will be used later.
We will create two subclasses of tabular_q_agent class to make the games work. 

Now you have to choose a game that uses only discrete observation spaces and discrete action spaces. These spaces must be represented by an Integer or by an array of Integers. For example, spaces like Boxes, Matrices and MultiArrays are NOT allowed. 

You can watch the class env file in order to verify this fact: you have to go in the right subfolder of the game contained in the envs folder and check if the obs_space and the action_space can be connected with DiscreteEnv class in discrete.py, which is contained in the same folder of the environment class file. It's important to specify that DiscreteEnv class refers to Discrete class: this file, which can be found in a gym folder named "spaces", describes discrete space features.

I've chosen for you two games that follow these specs: "Taxi" and "Roulette", contained in envs/toy_text.

### TAXI GAME:

Q-learning implementation of 'Taxi-v2'. In gym/spaces folder you can find the space descriptor file, whereas in gym/envs/toy_text folder you can use the other two scripts (taxi.py, discrete.py), describing the specs of the environment.

You must write a script that creates the simulation's env, concerning the class of the game you have chosen.
In my case: env=taxi.TaxiEnv() to create a taxi game env.

Then you have to create the Q-learning agent providing the observation and action space from env object you have just created. TabularQAgentTaxi class of tabular_q_agent_taxi.py checks if parameters are not correct, communicating it to the user. During this agent's creation you can also specify other q-learning details as, for example, the mean, the standard deviation, the greedy eps etc.

Finally you have to run (with trials) the learn function in this way: agent.learn(name_of_env).
This instrucion allows the agent to act properly on the env.

I've added the plot command from matplotlib to print on screen a plot of rewards recollected during the trials. I've created a list of trials, a list of rewards (read from a file). In addiction, the command plt.plot(eps,newRewards) generates a plot that will be printed on screen. 

### ROULETTE GAME:

Q-learning implementation of 'Roulette-v0'. In gym/spaces folder you can find the space descriptor file, whereas in gym/envs/toy_text folder you can use the other two scripts (roulette.py, discrete.py), describing the specs of the environment.

From now on, you have to follow the same syntax instructions of the first game.

N.B. I've used the softmax eps code because the random action is not useful for this game.

### SARSA VERSION FOR THE GAMES:

I've added Sarsa Algorithm for both games. The code is almost the same. The only difference is the change to the learn() function because at the beginning is chosen the first action. Then the action is updated at every step until the game reaches the terminal state.

### Q-LEARNING VERSION AND SARSA VERSION WITH COMPARISON: CONSIDERATIONS

I've added Sarsa implementation of Q-learning for both games. In this version, the learn function updates action at every step.
We have the opportunity to compare Sarsa and tabular Q-learning rewards plots.

Sarsa Algortihm works a bit better in most of the cases. This is because the actions are not always selected from greedy eps policy.

In the folder named "plots" there are the an example of comparison plots for roulette and taxi game on 1000 trials.

### IMPORTANT CHANGES TO THE ORIGINAL tabular_q_agent.py BEFORE RUNNING THE CODE:

In taxi and roulette cases I've modified tabularQAgent class into tabular_q_agent_taxi and tabular_q_agent_roulette (These classes are subclasses of tabular_q_agent class):

1. 	I've added libraries to make the code work.
2. 	I've replaced self.q[obs.items()] with self.q[obs] because both envs have an Integer as obs descriptor instead of an Integers type array.
3.	I've calculated the arithmetic mean of the rewards after each trial and I've saved it on an external file. 
	We will use this file to do the rewards' plot.
4.	In both games, I've used the soft greedy eps because less iterations are required to see good results with this eps.
	In the second game I've added two functions that implement softmax eps action instead of random action.
5.	Remember that every game has it own rules. You might have to modify the code like me to make the game work. 
6.	I've added a function LogUpdate() that creates a file log.txt to save some stats.

### How to run the code:

Both files (Taxi.py, tabular_q_agent.py tabular_q_agent_taxi.py and Roulette.py and tabular_q_agent_roulette.py) for both games must be in the same folder.

(follow the same steps to run the Sarsa implementation)

- To shell: python2.7 Taxi.py 
- To shell: python2.7 Roulette.py
- In the end you can save the graph on your pc running to shell: python2.7 print_taxi_rewards_plot.py or print_roulette_rewards_plot.py 

For comparison plots:
- To shell: python2.7 compare_sarsa_q_learning_taxi.py or compare_sarsa_q_learning_roulette.py
- After iterations you can run to shell: python2.7 print_taxi_comparison_plots.py or print_roulette_comparison_plots.py to see the plots

### Authors:

* Giacomo Ferro - https://github.com/JackIron

### License:

* MIT license

### Acknowledgments: 

* tabular_q_agent.py Former Developers 
* softmax eps Former Developers
