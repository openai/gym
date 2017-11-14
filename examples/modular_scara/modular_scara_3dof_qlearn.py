import gym
import gym_gazebo
from qlearn import QLearn
import time
import numpy
import pandas
from itertools import product
from functools import reduce
import tensorflow as tf
from tensorboard_logging import Logger

# Inspired by Basic Reinforcement Learning Tutorial 4: Q-learning in OpenAI gym
#  https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial4/q-learning-gym-1.py

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]

def step(action, state):
    """
    Implementation of "step" which uses the given action
    as an offset of the existing state. This function overloads
    the environment step method and should be used when actions-as-offsets
    are required for this particular environment.
    """
    # if action == (0.0, 0.0, 0.0):
        # print("\t\trobot decided to stay still!")
    offset_action = [a + s for a,s in zip(list(action), state)]
    # print("step: action: ", action)
    # print("step: state: ", state)
    # print("step: offset_action: ", offset_action)
    observation, reward, done, info = env.step(offset_action)
    return observation, reward, done, info


# Create the environment
env = gym.make('GazeboModularScara3DOF-v2')

# print(env.action_space) # Box(3,)
# print(env.observation_space) # Box(9,)
# print(env.observation_space.shape) # (9,)
# print(env.observation_space.shape[0]) # 9
# print(env.action_space) # 9

goal_average_steps = 2
max_number_of_steps = 20
last_time_steps = numpy.ndarray(0)
n_bins = 10
epsilon_decay_rate = 0.99 ########
save_model_with_prefix = 'qlearn_test' ######
it = 1 ######

number_of_features = env.observation_space.shape[0] # 9 however we're only using 3 of them (joint angles)
last_time_steps = numpy.ndarray(0)

# Number of states is huge so in order to simplify the situation
# typically, we discretize the space to: n_bins ** number_of_features
joint1_bins = pandas.cut([-numpy.pi/2, numpy.pi/2], bins=n_bins, retbins=True)[1][1:-1]
joint2_bins = pandas.cut([-numpy.pi/2, numpy.pi/2], bins=n_bins, retbins=True)[1][1:-1]
joint3_bins = pandas.cut([-numpy.pi/2, numpy.pi/2], bins=n_bins, retbins=True)[1][1:-1]
action_bins = pandas.cut([-numpy.pi/2, numpy.pi/2], bins=n_bins, retbins=True)[1][1:-1]
# print("joint1_bins: ", joint1_bins)

difference_bins = abs(joint1_bins[0] - joint1_bins[1])
# print("difference_bins: ", difference_bins)

# Generate actions
############
# test actions
# actions = [(0.0, 0.0, 0.0), (numpy.pi/2, numpy.pi/2, numpy.pi/2), (0 , 0, numpy.pi/2)]
# actions = [(0.0, 0.0, 0.0), (numpy.pi/2, numpy.pi/2, numpy.pi/2), (0, 0, numpy.pi/5)]

# generate actions so that it can re #####ach any "measurable" point in the space from anywhere
# with 10 bins and combinations of 3 elements: 729 possible actions.
# actions = []
# for x in product(action_bins, repeat=3):
#     actions.append(x)

# the following actions represent an offset over the existing state,
#  modifications over the traditional "step" method of the environment
#  are implemented in the "step" function
actions = [(difference_bins, 0.0, 0.0), (-difference_bins, 0.0, 0.0),
            (0.0, difference_bins, 0.0), (0.0, -difference_bins, 0.0),
            (0.0, 0.0, difference_bins), (0.0, 0.0, -difference_bins),
            (0.0, 0.0, 0.0)]
############

# The Q-learn algorithm
#qlearn = QLearn(actions=actions,
#    alpha=0.2, gamma=0.90, epsilon=0.5, epsilon_decay_rate=0.99)

qlearn = QLearn(actions=actions,
    alpha=0.5, gamma=0.30, epsilon=0.8, epsilon_decay_rate=0.999)

# logs_path = "/tmp/modular_scara_3dof_qlearn/"
logger = Logger("/tmp/rosrl/GazeboModularScara3DOF-v2/q-learn/")

with tf.Session() as sess:
    # op to write logs to Tensorboard
    # summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    for i_episode in range(1000): # episodes
        print("I_EPISODE", i_episode)#####
        observation = env.reset()

        joint1_position, joint2_position, joint3_position  = observation[:3]
        state = build_state([to_bin(joint1_position, joint1_bins),
                         to_bin(joint2_position, joint2_bins),
                         to_bin(joint3_position, joint3_bins)])
        episode_reward = 0

        for t in range(max_number_of_steps):
            env.render()
            # print("join1_bins", joint1_bins)
            # print("Number of steps", t)
            # print("\t\tq: ",qlearn.q)
            # Pick an action based on the current f
            # print("state: ",state)
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # print("state:", state)##
            # print("action: ",action)
            # print("state observation: ",observation[:3])###

            # Execute the action and get feedback
            # observation, reward, done, info = env.step(action) # environment step method
            observation, reward, done, info = step(action, observation[:3])   # local step method, applies the action as an offset
                                                                                # to the state
            # print("reward: ",reward)
            episode_reward += reward
            # print("next state observation: ",observation[:3])###

            # Digitize the observation to get a state
            joint1_position, joint2_position, joint3_position  = observation[:3]
            nextState = build_state([to_bin(joint1_position, joint1_bins),
                            to_bin(joint2_position, joint2_bins),
                            to_bin(joint3_position, joint3_bins)])

            # print("nextState", nextState)
            # summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward",simple_value=episode_reward)])
            # summary_writer.add_summary(summary, i_episode * max_number_of_steps + t)

            if done:
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
                break
            else:
                # Q-learn stuff
                #qlearn.learn(state, action, reward, nextState)
                qlearn.learn(state, action, reward, nextState, save_model_with_prefix, it)
                state = nextState

                it += 1 #####

        logger.log_scalar("episode_reward", episode_reward, i_episode)
        print("episode reward: ",episode_reward)

# l = last_time_steps.tolist()
# l.sort()
# print("Overall score: {:0.2f}".format(last_time_steps.mean()))
# print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))
