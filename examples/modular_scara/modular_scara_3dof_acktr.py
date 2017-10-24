import gym
import gym_gazebo
import tensorflow as tf
import argparse
import copy
import sys

# Use algorithms from baselines
from baselines.acktr.acktr_cont import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction
from baselines.common import set_global_seeds

def train(env, num_timesteps, seed, save_model_with_prefix, restore_model_from_file):
    """
    Train an SCARA Agent using ACKTR technique
        inputs:
            - env: environment
            - num_timesteps: number of steps of the robot will take to "learn", how much it'll run
            - seed: seed of the environment
            - save_model_with_prefix: experiments/<save_model_with_prefix>/<save_model_with_prefix>_iteration whereto save the models
            - restore_model_from_file: model to restore
    """
    set_global_seeds(seed)
    env.seed(seed)

    with tf.Session(config=tf.ConfigProto()) as session:
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        learn(env,
            policy=policy, vf=vf,
            gamma=0.99,
            lam=0.97,
            timesteps_per_batch=2500,
            desired_kl=0.02,
            num_timesteps=num_timesteps,
            animate=False,
            save_model_with_prefix=save_model_with_prefix,
            restore_model_from_file=restore_model_from_file)

env = gym.make('GazeboModularScara3DOF-v0')
initial_observation = env.reset()
print("Initial observation: ", initial_observation)
env.render()

train(env,
        num_timesteps=1e6,
        seed=0, # default value
        save_model_with_prefix='',
        restore_model_from_file='')
