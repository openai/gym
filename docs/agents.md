# Agents

An "agent" describes the method of running an RL algorithm against an environment in the gym. The agent may contain the algorithm itself or simply provide an integration between an algorithm and the gym environments. Submit another to this list via a pull-request. 

_**NOTICE**: Evaluations submitted to the scoreboard are encouraged to link a writeup (gist) about duplicating the results. These writeups will likely direct you to specific algorithms. This agent listing is not attempting to replace writeups and will likely in time be filled with general purpose agents that will serve as a great starting place for those looking for tooling integrations or general algorithm ideas and attempts._

## RandomAgent

A sample agent located in this repo at `gym/examples/agents/random_agent.py`. This simple agent leverages the environments ability to produce a random valid action and does so for each step.  

## cem.py

A generic Cross-Entropy agent located in this repo at `gym/examples/agents/cem.py`. This agent defaults to 10 iterations of 25 episodes considering the top 20% "elite".

## TabularQAgent

Agent implementing tabular Q-learning located in this repo at `gym/examples/agents/tabular_q_agent.py`. 

## dqn

This is a very basic DQN (with experience replay) implementation, which uses OpenAI's gym environment and Keras/Theano neural networks. [/sherjilozair/dqn](https://github.com/sherjilozair/dqn)

## rllab

a framework for developing and evaluating reinforcement learning algorithms, fully compatible with OpenAI Gym. It includes a wide range of continuous control tasks plus implementations of many algorithms. [/rllab/rllab](https://github.com/rllab/rllab)