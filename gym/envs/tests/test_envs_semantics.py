import numpy as np
import json
import hashlib
import os

import logging
logger = logging.getLogger(__name__)

import gym
from gym import envs, spaces

from generate_json import create_rollout

DATA_DIR = './gym/envs/tests/rollout_data/'
ROLLOUT_FILE = DATA_DIR + 'rollout-filenames.json'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.isfile(ROLLOUT_FILE): 
  with open(ROLLOUT_FILE, "w") as outfile:
    json.dump({}, outfile, indent=2)

def test_env_semantics(spec):
  with open(ROLLOUT_FILE) as data_file:
    rollout_filenames = json.load(data_file)

  if spec.id not in rollout_filenames:
    added = create_rollout(spec)
    if added: 
      with open(ROLLOUT_FILE) as data_file:
        rollout_filenames = json.load(data_file)
    else:
      return

  with open(rollout_filenames[spec.id]) as data_file:
    data = json.load(data_file)

  logger.info("Testing rollout with {} actions for {} environment...".format(len(data), spec.id))

  # Set same seeds as set in generate_json.py 
  spaces.seed(0)
  env = spec.make()
  env.seed(0)
  observation_now = env.reset()

  num_passed = 0
  for (action, observation, reward, done) in data:
    action = env.action_space.from_jsonable(action)[0]

    observation_now, reward_now, done_now, _ = env.step(action)
    observation_now = hashlib.sha1(str(observation_now)).hexdigest()

    assert observation == observation_now, 'Observation not equal for action number {}'.format(num_passed)
    assert reward == str(reward_now), 'Reward not equal for action number {}'.format(num_passed)
    assert done == str(done_now), 'Done not equal for action number {}'.format(num_passed)

    num_passed += 1

    if done == 'True':
      observation_now = env.reset()

def test_all_env_semantics():
  specs = [spec for spec in envs.registry.all() if spec._entry_point is not None]

  for spec in specs: 
    test_env_semantics(spec)


test_all_env_semantics()
