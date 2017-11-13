import numpy as np
import sys

import gym
import gym_gazebo

import tensorflow as tf

import argparse
import copy

from baselines import logger
from baselines.common import set_global_seeds, tf_util as U

from baselines.acktr.acktr_cont import learn
from baselines.agent.utility.general_utils import get_ee_points, get_position
from baselines.ppo1 import mlp_policy, pposgd_simple


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
def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
    hid_size=64, num_hid_layers=2)
# gym.logger.setLevel(logging.WARN)
obs = env.reset()
print("Initial obs: ", obs)
# env.seed(seed)
pi = policy_fn('pi', env.observation_space, env.action_space)
tf.train.Saver().restore(sess, '/home/rkojcev/devel/baselines/baselines/experiments/ros1_ppo1_test_H/saved_models/ros1_ppo1_test_H_afterIter_424.model')
done = False
while True:
    action = pi.act(True, obs)[0]
    obs, reward, done, info = env.step(action)
    print(action)
