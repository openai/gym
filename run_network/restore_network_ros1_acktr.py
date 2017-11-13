import numpy as np
import sys

import gym
import gym_gazebo

import tensorflow as tf

import argparse
import copy

from baselines import logger
from baselines.common import set_global_seeds, tf_util as U

# Use algorithms from baselines
from baselines.acktr.acktr_cont import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction
from baselines.common import set_global_seeds


env = gym.make('GazeboModularScara3DOF-v3')
initial_observation = env.reset()
print("Initial observation: ", initial_observation)
env.render()
seed = 0
parser = argparse.ArgumentParser(description='Run Gazebo benchmark.')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--save_model_with_prefix',
                            help='Specify a prefix name to save the model with after every iters. Note that this will generate multiple files (*.data, *.index, *.meta and checkpoint) with the same prefix', default='')
parser.add_argument('--restore_model_from_file',
                            help='Specify the absolute path to the model file including the file name upto .model (without the .data-00000-of-00001 suffix). make sure the *.index and the *.meta files for the model exists in the specified location as well', default='')
args = parser.parse_args()

sess = U.make_session(num_cpu=1)
sess.__enter__()
# logger.session().__enter__()

# with tf.Session(config=tf.ConfigProto()) as session:
obs = []
acs = []
ac_dists = []
logps = []
rewards = []
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.shape[0]
ob = env.reset()
prev_ob = np.float32(np.zeros(ob.shape))
state = np.concatenate([ob, prev_ob], -1)
obs.append(state)
with tf.variable_scope("vf"):
    vf = NeuralNetValueFunction(ob_dim, ac_dim)
with tf.variable_scope("pi"):
    policy = GaussianMlpPolicy(ob_dim, ac_dim)
tf.train.Saver().restore(sess, '/home/rkojcev/baselines_networks/ros1_acktr_H/saved_models/ros1_acktr_H_afterIter_267.model')
done = False
ac, ac_dist, logp = policy.act(state)
# print("action: ", ac)
acs.append(ac)
ac_dists.append(ac_dist)
logps.append(logp)
prev_ob = np.copy(ob)

while True:
    ac, ac_dist, logp = policy.act(state)
    # here I need to figure out how to take non-biased action.
    scaled_ac = env.action_space.low + (ac + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
    # scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)
    scaled_ac = np.clip(scaled_ac, None, env.action_space.high)
    ob, rew, done, _ = env.step(scaled_ac)
