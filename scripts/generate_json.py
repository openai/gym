from __future__ import unicode_literals
from gym import envs, spaces
import json
import os
import sys
import hashlib
import argparse

import logging
logger = logging.getLogger(__name__)

from gym.envs.tests.spec_list import should_skip_env_spec_for_tests
from gym.envs.tests.test_envs_semantics import generate_rollout_hash, hash_object

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'gym', 'envs', 'tests')
ROLLOUT_STEPS = 100
episodes = ROLLOUT_STEPS
steps = ROLLOUT_STEPS

ROLLOUT_FILE = os.path.join(DATA_DIR, 'rollout.json')

if not os.path.isfile(ROLLOUT_FILE):
  logger.info("No rollout file found. Writing empty json file to {}".format(ROLLOUT_FILE))
  with open(ROLLOUT_FILE, "w") as outfile:
    json.dump({}, outfile, indent=2)

def update_rollout_dict(spec, rollout_dict):
  """
  Takes as input the environment spec for which the rollout is to be generated,
  and the existing dictionary of rollouts. Returns True iff the dictionary was
  modified.
  """
  # Skip platform-dependent
  if should_skip_env_spec_for_tests(spec):
    logger.info("Skipping tests for {}".format(spec.id))
    return False

  # Skip environments that are nondeterministic
  if spec.nondeterministic:
    logger.info("Skipping tests for nondeterministic env {}".format(spec.id))
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

  existing = rollout_dict.get(spec.id)
  if existing:
    differs = False
    for key, new_hash in rollout.items():
      differs = differs or existing[key] != new_hash
    if not differs:
      logger.debug("Hashes match with existing for {}".format(spec.id))
      return False
    else:
      logger.warn("Got new hash for {}. Overwriting.".format(spec.id))

  rollout_dict[spec.id] = rollout
  return True

def add_new_rollouts(spec_ids, overwrite):
  environments = [spec for spec in envs.registry.all() if spec._entry_point is not None]
  if spec_ids:
    environments = [spec for spec in environments if spec.id in spec_ids]
    assert len(environments) == len(spec_ids), "Some specs not found"
  with open(ROLLOUT_FILE) as data_file:
    rollout_dict = json.load(data_file)
  modified = False
  for spec in environments:
    if not overwrite and spec.id in rollout_dict:
      logger.debug("Rollout already exists for {}. Skipping.".format(spec.id))
    else:
      modified = update_rollout_dict(spec, rollout_dict) or modified

  if modified:
    logger.info("Writing new rollout file to {}".format(ROLLOUT_FILE))
    with open(ROLLOUT_FILE, "w") as outfile:
      json.dump(rollout_dict, outfile, indent=2, sort_keys=True)
  else:
    logger.info("No modifications needed.")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--force', action='store_true', help='Overwrite '+
    'existing rollouts if hashes differ.')
  parser.add_argument('-v', '--verbose', action='store_true')
  parser.add_argument('specs', nargs='*', help='ids of env specs to check (default: all)')
  args = parser.parse_args()
  if args.verbose:
    logger.setLevel(logging.DEBUG)
  add_new_rollouts(args.specs, args.force)
