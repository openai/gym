from __future__ import unicode_literals
import json
import hashlib
import os
import sys
import logging
import pytest
logger = logging.getLogger(__name__)
from gym import envs, spaces
from gym.envs.tests.spec_list import spec_list

DATA_DIR = os.path.dirname(__file__)
ROLLOUT_STEPS = 100
episodes = ROLLOUT_STEPS
steps = ROLLOUT_STEPS

ROLLOUT_FILE = os.path.join(DATA_DIR, 'rollout.json')

if not os.path.isfile(ROLLOUT_FILE):
  with open(ROLLOUT_FILE, "w") as outfile:
    json.dump({}, outfile, indent=2)

def hash_object(unhashed):
  return hashlib.sha256(str(unhashed).encode('utf-16')).hexdigest()

def generate_rollout_hash(spec):
  spaces.seed(0)
  env = spec.make()
  env.seed(0)

  observation_list = []
  action_list = []
  reward_list = []
  done_list = []

  total_steps = 0
  for episode in range(episodes):
    if total_steps >= ROLLOUT_STEPS: break
    observation = env.reset()

    for step in range(steps):
      action = env.action_space.sample()
      observation, reward, done, _ = env.step(action)

      action_list.append(action)
      observation_list.append(observation)
      reward_list.append(reward)
      done_list.append(done)

      total_steps += 1
      if total_steps >= ROLLOUT_STEPS: break

      if done: break

  observations_hash = hash_object(observation_list)
  actions_hash = hash_object(action_list)
  rewards_hash = hash_object(reward_list)
  dones_hash = hash_object(done_list)

  return observations_hash, actions_hash, rewards_hash, dones_hash

@pytest.mark.parametrize("spec", spec_list)
def test_env_semantics(spec):
  with open(ROLLOUT_FILE) as data_file:
    rollout_dict = json.load(data_file)

  if spec.id not in rollout_dict:
    if not spec.nondeterministic:
      logger.warn("Rollout does not exist for {}, run generate_json.py to generate rollouts for new envs".format(spec.id))
    return

  logger.info("Testing rollout for {} environment...".format(spec.id))

  observations_now, actions_now, rewards_now, dones_now = generate_rollout_hash(spec)

  errors = []
  if rollout_dict[spec.id]['observations'] != observations_now:
    errors.append('Observations not equal for {} -- expected {} but got {}'.format(spec.id, rollout_dict[spec.id]['observations'], observations_now))
  if rollout_dict[spec.id]['actions'] != actions_now:
    errors.append('Actions not equal for {} -- expected {} but got {}'.format(spec.id, rollout_dict[spec.id]['actions'], actions_now))
  if rollout_dict[spec.id]['rewards'] != rewards_now:
    errors.append('Rewards not equal for {} -- expected {} but got {}'.format(spec.id, rollout_dict[spec.id]['rewards'], rewards_now))
  if rollout_dict[spec.id]['dones'] != dones_now:
    errors.append('Dones not equal for {} -- expected {} but got {}'.format(spec.id, rollout_dict[spec.id]['dones'], dones_now))
  if len(errors):
    for error in errors:
      logger.warn(error)
    raise ValueError(errors)
