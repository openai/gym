import gym
from gym import envs, spaces 
import json
import numpy as np
import datetime
import os

from test_envs import should_skip_env_spec_for_tests

DATA_DIR = './rollout_data/'
ROLLOUT_STEPS = 25
episodes = ROLLOUT_STEPS
steps = ROLLOUT_STEPS

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

environments = [spec for spec in envs.registry.all() if spec._entry_point is not None]

json_file_dict = {}

for spec in environments:

  if should_skip_env_spec_for_tests(spec):
    continue

  # Skip environments that are nondeterministic
  if spec.nondeterministic:
    print "Skipping tests for", spec.id
    continue

  # Temporarily skip Doom environments until setup issues resolved
  if 'Doom' in spec.id:
    print "Skipping tests for", spec.id
    continue

  print "Generating rollout for {}".format(spec.id)

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

      #rollout.append((np.array(action).tolist(), np.array(observation).tolist(), str(reward), str(done)))

      action = env.action_space.to_jsonable(action)
      observation = env.observation_space.to_jsonable(observation)

      rollout.append((action, observation, str(reward), str(done)))

      total_steps += 1
      if total_steps >= ROLLOUT_STEPS: break

      if done: break

  date = datetime.datetime.now().date()

  filename = DATA_DIR + str(date) + '-' + spec.id + '-rollout.json'
  json_file_dict[spec.id] = filename
  with open(filename, "w") as outfile:
    json.dump(rollout, outfile, indent=2)


with open(DATA_DIR + str(date) + '-rollout-filenames.json', "w") as outfile:
  json.dump(json_file_dict, outfile, indent=2)




