import numpy as np
from nose2 import tools
import os

import logging
logger = logging.getLogger(__name__)

import gym
from gym import envs

def should_skip_env_spec_for_tests(spec):
    # We skip tests for envs that require dependencies or are otherwise
    # troublesome to run frequently

    # Skip mujoco tests for pull request CI
    skip_mujoco = not (os.environ.get('MUJOCO_KEY_BUNDLE') or os.path.exists(os.path.expanduser('~/.mujoco')))
    if skip_mujoco and spec._entry_point.startswith('gym.envs.mujoco:'):
        return True

    # TODO(jonas 2016-05-11): Re-enable these tests after fixing box2d-py
    if spec._entry_point.startswith('gym.envs.box2d:'):
        logger.warn("Skipping tests for box2d env {}".format(spec._entry_point))
        return True

    # Skip ConvergenceControl tests (the only env in parameter_tuning) according to pull #104
    if spec._entry_point.startswith('gym.envs.parameter_tuning:'):
        logger.warn("Skipping tests for parameter_tuning env {}".format(spec._entry_point))
        return True

    # Skip Semisuper tests for now (broken due to monitor refactor)
    if spec._entry_point.startswith('gym.envs.safety:Semisuper'):
        logger.warn("Skipping tests for semisuper env {}".format(spec._entry_point))
        return True

    return False


# This runs a smoketest on each official registered env. We may want
# to try also running environments which are not officially registered
# envs.
specs = [spec for spec in sorted(envs.registry.all(), key=lambda x: x.id) if spec._entry_point is not None]
@tools.params(*specs)
def test_env(spec):
    if should_skip_env_spec_for_tests(spec):
        return

    env = spec.make()
    ob_space = env.observation_space
    act_space = env.action_space
    ob = env.reset()
    assert ob_space.contains(ob), 'Reset observation: {!r} not in space'.format(ob)
    a = act_space.sample()
    observation, reward, done, _info = env.step(a)
    assert ob_space.contains(observation), 'Step observation: {!r} not in space'.format(observation)
    assert np.isscalar(reward), "{} is not a scalar for {}".format(reward, env)
    assert isinstance(done, bool), "Expected {} to be a boolean".format(done)

    for mode in env.metadata.get('render.modes', []):
        env.render(mode=mode)
    env.render(close=True)

    # Make sure we can render the environment after close.
    for mode in env.metadata.get('render.modes', []):
        env.render(mode=mode)
    env.render(close=True)

    env.close()

# Run a longer rollout on some environments
def test_random_rollout():
    for env in [envs.make('CartPole-v0'), envs.make('FrozenLake-v0')]:
        agent = lambda ob: env.action_space.sample()
        ob = env.reset()
        for _ in range(10):
            assert env.observation_space.contains(ob)
            a = agent(ob)
            assert env.action_space.contains(a)
            (ob, _reward, done, _info) = env.step(a)
            if done: break

def test_double_close():
    class TestEnv(gym.Env):
        def __init__(self):
            self.close_count = 0

        def _close(self):
            self.close_count += 1

    env = TestEnv()
    assert env.close_count == 0
    env.close()
    assert env.close_count == 1
    env.close()
    assert env.close_count == 1
