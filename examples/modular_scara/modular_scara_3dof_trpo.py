import gym
import gym_gazebo
import tensorflow as tf
import argparse
import copy
import sys
import numpy as np
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds, tf_util as U

from baselines.agent.utility.general_utils import get_ee_points, get_position
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
from baselines.trpo_mpi import trpo_mpi

import baselines.common.tf_util as U

env = gym.make('GazeboModularScara3DOF-v3')
initial_observation = env.reset()
print("Initial observation: ", initial_observation)
env.render()
seed = 0

sess = U.single_threaded_session()
sess.__enter__()

rank = MPI.COMM_WORLD.Get_rank()
if rank != 0:
    logger.set_level(logger.DISABLED)
workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
set_global_seeds(workerseed)
def policy_fn(name, ob_space, ac_space):
    return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
        hid_size=32, num_hid_layers=2)
env.seed(workerseed)
print(workerseed)

trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
            max_timesteps=1e6, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, save_model_with_prefix='ros1_trpo_test_H')
