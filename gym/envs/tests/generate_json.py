import gym
from gym import envs, spaces 
import json
import numpy as np
import datetime
import os
import hashlib

from test_envs import should_skip_env_spec_for_tests

DATA_DIR = './rollout_data/'
ROLLOUT_FILE = DATA_DIR + 'rollout-filenames.json'
ROLLOUT_STEPS = 100
episodes = ROLLOUT_STEPS
steps = ROLLOUT_STEPS

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.isfile(ROLLOUT_FILE): 
  with open(ROLLOUT_FILE, "w") as outfile:
    json.dump({}, outfile, indent=2)

def create_rollout(spec):
  """
  Takes as input the environment spec for which the rollout is to be generated.
  Returns a bool which indicates whether the new rollout was added to the json file.  

  """

  filename = DATA_DIR + spec.id + '-rollout.json'

  if should_skip_env_spec_for_tests(spec):
    return False

  # Skip environments that are nondeterministic
  if spec.nondeterministic:
    print "Skipping tests for", spec.id
    return False

  # Temporarily skip Doom environments until setup issues resolved
  if 'Doom' in spec.id:
    print "Skipping tests for", spec.id
    return False

  # Skip broken environments
  # TODO: look into these environments
  if spec.id in ['InterpretabilityCartpoleObservations-v0']:
    print "Skipping tests for", spec.id
    return False

  with open(ROLLOUT_FILE) as data_file:
    rollout_filenames = json.load(data_file)

  # Skip generating rollouts that already exist
  if spec.id in rollout_filenames:
    print "Rollout already exists for", spec.id
    return False   

  print "Generating rollout for {}".format(spec.id)

  rollout_filenames[spec.id] = filename

  with open(ROLLOUT_FILE, "w") as outfile:
    json.dump(rollout_filenames, outfile, indent=2)

  spaces.seed(0)
  env = spec.make()
  env.seed(0)

  rollout = []

  total_steps = 0
  for episode in xrange(episodes):
    if total_steps >= ROLLOUT_STEPS: break
    observation = env.reset()

    for step in xrange(steps):
      action = env.action_space.sample()
      observation, reward, done, _ = env.step(action)

      action = env.action_space.to_jsonable([action])
      observation = hashlib.sha1(str(observation)).hexdigest()

      rollout.append((action, observation, str(reward), str(done)))

      total_steps += 1
      if total_steps >= ROLLOUT_STEPS: break

      if done: break

  with open(filename, "w") as outfile:
    json.dump(rollout, outfile, indent=2)

  return True

def create_all_rollouts():
  environments = [spec for spec in envs.registry.all() if spec._entry_point is not None]

  for spec in environments:
    create_rollout(spec)
