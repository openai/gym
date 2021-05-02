import gym
import numpy as np
import random
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

## Defining the simulation related constants
NUM_EPISODES = 6000
goal_steps = 200
score_requirement = -198
intial_games = 10000

def state_to_bucket(state, state_bounds, num_buckets):
    #state comes as a tuple of 4 values: (cart position, cart velocity, pole angle, pole velocity at tip)
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= state_bounds[i][0]:
            bucket_index = 0
        elif state[i] >= state_bounds[i][1]:
            bucket_index = num_buckets[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = state_bounds[i][1] - state_bounds[i][0]
            offset = (num_buckets[i]-1)*state_bounds[i][0]/bound_width
            scaling = (num_buckets[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)

    return tuple(bucket_indice)


def select_action(state, explore_rate, env, q_table):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action

def model_data_preparation():
    training_data = []
    accepted_scores = []
    for game_index in range(intial_games):
        score = 0
        game_memory = []
        previous_observation = []
        for step_index in range(goal_steps):
            action = random.randrange(0, 3)
            observation, reward, done, info = env.step(action)
            
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
                
            previous_observation = observation
            if observation[0] > -0.2:
                reward = 1
            
            score += reward
            if done:
                break
            
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1, 0]
                elif data[1] == 0:
                    output = [1, 0, 0]
                elif data[1] == 2:
                    output = [0, 0, 1]
                training_data.append([data[0], output])
        
        env.reset()
    
    print(accepted_scores)
    
    return training_data

def simulate(starting_epsilon):
    ## Initialize the "Cart-Pole" environment

    env = gym.make('MountainCar-v0')

    num_buckets = (20, 15)  # (position, velocity)
    num_actions = env.action_space.n  # (left, right)

    # Bounds for each discrete state
    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))

    ## Creating a Q-Table for each state-action pair
    q_table = np.zeros(num_buckets + (num_actions,))

    ## Instantiating the learning related parameters
    learning_rate = .5
    explore_rate = starting_epsilon
    discount_factor = 0.9

    for episode in range(NUM_EPISODES):

        done = False
        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv, state_bounds, num_buckets)
        total_reward = 0
        steps = 0
        while done != True:
            if episode == NUM_EPISODES -1:
                env.render()

            # Select an action
            action = select_action(state_0, explore_rate, env, q_table)

            # Execute the action
            obv, reward, done, _ = env.step(action)

            total_reward += reward
            # Observe the result
            state = state_to_bucket(obv, state_bounds, num_buckets)

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate*(reward + discount_factor*(best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            steps +=1

        # reduce the eploration rate for the next episode
        reduction = starting_epsilon / NUM_EPISODES
        explore_rate -= reduction
        if episode%100 == 0:
            print("Episode", episode, "finished after", steps, "time steps with total reward", total_reward, "explore_rate", explore_rate)


if __name__ == "__main__":
    simulate(starting_epsilon=.1)

