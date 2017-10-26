import gym
import gym_gazebo
from qlearn import QLearn
import time
import numpy
import pandas

# Inspired by Basic Reinforcement Learning Tutorial 4: Q-learning in OpenAI gym
#  https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial4/q-learning-gym-1.py

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]

# Create the environment
env = gym.make('GazeboModularScara3DOF-v2')

# print(env.action_space) # Box(3,)
# print(env.observation_space) # Box(9,)
# print(env.observation_space.shape) # (9,)
# print(env.observation_space.shape[0]) # 9
# print(env.action_space) # 9

goal_average_steps = 2
max_number_of_steps = 10
last_time_steps = numpy.ndarray(0)
n_bins = 10

number_of_features = env.observation_space.shape[0] # 9 however we're only using 3 of them (joint angles)
last_time_steps = numpy.ndarray(0)

# Number of states is huge so in order to simplify the situation
# typically, we discretize the space to: n_bins ** number_of_features
joint1_bins = pandas.cut([-numpy.pi/2, numpy.pi/2], bins=n_bins, retbins=True)[1][1:-1]
joint2_bins = pandas.cut([-numpy.pi/2, numpy.pi/2], bins=n_bins, retbins=True)[1][1:-1]
joint3_bins = pandas.cut([-numpy.pi/2, numpy.pi/2], bins=n_bins, retbins=True)[1][1:-1]

# Generate posible actions
# TODO program this
actions = [item for innerlist in outerlist ]

# The Q-learn algorithm
qlearn = QLearn(actions=actions,
    alpha=0.5, gamma=0.90, epsilon=0.1)

for i_episode in range(30): # episodes
    observation = env.reset()

    joint1_position, joint2_position, joint3_position  = observation[:3]
    state = build_state([to_bin(joint1_position, joint1_bins),
                     to_bin(joint2_position, joint2_bins),
                     to_bin(joint3_position, joint3_bins)])

    for t in range(max_number_of_steps):
        env.render()

        # Pick an action based on the current state
        action = qlearn.chooseAction(state)
        # Execute the action and get feedback
        observation, reward, done, info = env.step(action)

        # Digitize the observation to get a state
        joint1_position, joint2_position, joint3_position  = observation[:3]
        nextState = build_state([to_bin(joint1_position, joint1_bins),
                        to_bin(joint2_position, joint2_bins),
                        to_bin(joint3_position, joint3_bins)])

        if done:
            last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
            break
        else:
            # Q-learn stuff
            qlearn.learn(state, action, reward, nextState)
            state = nextState

l = last_time_steps.tolist()
l.sort()
print("Overall score: {:0.2f}".format(last_time_steps.mean()))
print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))
