from __future__ import unicode_literals
from gym import envs, spaces 
import json
import os
import sys
import hashlib

import logging
logger = logging.getLogger(__name__)

from gym.envs.tests.test_envs import should_skip_env_spec_for_tests
from gym.envs.tests.test_envs_semantics import generate_rollout_hash, hash_object

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'gym', 'envs', 'tests')
ROLLOUT_STEPS = 100
episodes = ROLLOUT_STEPS
steps = ROLLOUT_STEPS

python_version = sys.version_info.major
if python_version == 3:
    ROLLOUT_FILE = os.path.join(DATA_DIR, 'rollout_py3.json')
else:
    ROLLOUT_FILE = os.path.join(DATA_DIR, 'rollout_py2.json')


if not os.path.isfile(ROLLOUT_FILE): 
  with open(ROLLOUT_FILE, "w") as outfile:
    json.dump({}, outfile, indent=2)

def create_rollout(spec):
  """
  Takes as input the environment spec for which the rollout is to be generated.
  Returns a bool which indicates whether the new rollout was added to the json file.  

  """
  # Skip platform-dependent Doom environments
  if should_skip_env_spec_for_tests(spec) or 'Doom' in spec.id:
    logger.warn("Skipping tests for {}".format(spec.id))
    return False

  # Skip environments that are nondeterministic
  if spec.nondeterministic:
    logger.warn("Skipping tests for nondeterministic env {}".format(spec.id))
    return False

  # Skip broken environments
  # TODO: look into these environments
  if spec.id in ['PredictObsCartpole-v0', 'InterpretabilityCartpoleObservations-v0']:
    logger.warn("Skipping tests for {}".format(spec.id))
    return False

  with open(ROLLOUT_FILE) as data_file:
    rollout_dict = json.load(data_file)

  # Skip generating rollouts that already exist
  if spec.id in rollout_dict:
    logger.warn("Rollout already exists for {}".format(spec.id))
    return False

  logger.info("Generating rollout for {}".format(spec.id))

  try:
    observations_hash, actions_hash, rewards_hash, dones_hash = generate_rollout_hash(spec)
  except:
    # If running the env generates an exception, don't write to the rollout file
    logger.warn("Exception {} thrown while generating rollout for {}. Rollout not added.".format(sys.exc_info()[0], spec.id))
    return False

  rollout = {}
  rollout['observations'] = observations_hash
  rollout['actions'] = actions_hash
  rollout['rewards'] = rewards_hash
  rollout['dones'] = dones_hash

  rollout_dict[spec.id] = rollout

  with open(ROLLOUT_FILE, "w") as outfile:
    json.dump(rollout_dict, outfile, indent=2)

  return True

def add_new_rollouts():
  environments = [spec for spec in envs.registry.all() if spec._entry_point is not None]

  for spec in environments:
    create_rollout(spec)

add_new_rollouts()
