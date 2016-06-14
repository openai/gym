import numpy as np
import json

import logging

import gym
from gym import envs, spaces

DATA_DIR = './rollout_data/'
ROLLOUT_FILE = DATA_DIR + '2016-06-11-rollout-filenames.json'

def test_env_semantics(spec):
  with open(ROLLOUT_FILE) as data_file:
    rollout_filenames = json.load(data_file)

  if spec.id not in rollout_filenames:
    print "Rollout not found for {} environment".format(spec.id)
    return

  # TODO: why is from_jsonable throwing errors for the Box action_space?
  if spec.id == 'Pendulum-v0':
    return

  with open(rollout_filenames[spec.id]) as data_file:
    data = json.load(data_file)

  print "Testing rollout with {} actions for {} environment...".format(len(data), spec.id)

  # Set same seeds as set in generate_json.py 
  spaces.seed(0)
  env = spec.make()
  env.seed(0)
  observation_now = env.reset()

  num_passed = 0
  for (action, observation, reward, done) in data:
    action = env.action_space.from_jsonable(action)
    observation = env.observation_space.from_jsonable(observation)

    observation_now, reward_now, done_now, _ = env.step(action)

    assert np.array_equal(observation, observation_now), 'Observation not equal for action number {}'.format(num_passed)
    assert np.array_equal(reward, str(reward_now)), 'Reward not equal for action number {}'.format(num_passed)
    assert np.array_equal(done, str(done_now)), 'Done not equal for action number {}'.format(num_passed)

    num_passed += 1

    if done == 'True':
      observation_now = env.reset()

def test_all_env_semantics():
  specs = [spec for spec in envs.registry.all() if spec._entry_point is not None and spec.id != 'ElevatorAction-v0']

  for spec in specs: 
    test_env_semantics(spec)

test_all_env_semantics()

