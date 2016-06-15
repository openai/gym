from gym import envs, spaces 
import json
import numpy as np
import datetime
import os
import hashlib

import logging
logger = logging.getLogger(__name__)

from gym.envs.tests.test_envs import should_skip_env_spec_for_tests

DATA_DIR = os.path.join(os.pardir, 'gym', 'envs', 'tests')
ROLLOUT_FILE = os.path.join(DATA_DIR, 'rollout_test.json')
ROLLOUT_STEPS = 100
episodes = ROLLOUT_STEPS
steps = ROLLOUT_STEPS

if not os.path.isfile(ROLLOUT_FILE): 
  with open(ROLLOUT_FILE, "w") as outfile:
    json.dump({}, outfile, indent=2)

def create_rollout(spec):
  """
  Takes as input the environment spec for which the rollout is to be generated.
  Returns a bool which indicates whether the new rollout was added to the json file.  

  """

  if should_skip_env_spec_for_tests(spec):
    logger.warn("Skipping tests for {}".format(spec.id))
    return False

  # Skip environments that are nondeterministic
  if spec.nondeterministic:
    logger.warn("Skipping tests for nondeterministic env {}".format(spec.id))
    return False

  # Skip broken environments
  # TODO: look into these environments
  if spec.id in ['PredictObsCartpole-v0']:
    logger.warn("Skipping tests for {}".format(spec.id))
    return False

  with open(ROLLOUT_FILE) as data_file:
    rollout_dict = json.load(data_file)

  # Skip generating rollouts that already exist
  if spec.id in rollout_dict:
    logger.warn("Rollout already exists for {}".format(spec.id))
    return False   

  logger.info("Generating rollout for {}".format(spec.id))

  spaces.seed(0)
  env = spec.make()
  env.seed(0)

  rollout = {}

  action_list = []
  observation_list = []
  reward_list = []
  done_list = []

  total_steps = 0
  for episode in xrange(episodes):
    if total_steps >= ROLLOUT_STEPS: break
    observation = env.reset()

    for step in xrange(steps):
      action = env.action_space.sample()
      observation, reward, done, _ = env.step(action)

      action_list.append(action)
      observation_list.append(observation)
      reward_list.append(reward)
      done_list.append(done)

      total_steps += 1
      if total_steps >= ROLLOUT_STEPS: break

      if done: break

  rollout['observations'] = hashlib.sha1(str(observation_list)).hexdigest()
  rollout['actions'] = hashlib.sha1(str(action_list)).hexdigest()
  rollout['rewards'] = hashlib.sha1(str(reward_list)).hexdigest()
  rollout['dones'] = hashlib.sha1(str(done_list)).hexdigest()

  rollout_dict[spec.id] = rollout

  with open(ROLLOUT_FILE, "w") as outfile:
    json.dump(rollout_dict, outfile, indent=2)

  return True

def add_new_rollouts():
  environments = [spec for spec in envs.registry.all() if spec._entry_point is not None]

  for spec in environments:
    create_rollout(spec)

add_new_rollouts()
