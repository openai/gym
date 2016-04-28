import numpy as np
from nose2 import tools
import os

from gym import envs

# This runs a smoketest on each official registered env. We may want
# to try also running environments which are not officially registered
# envs.
specs = [spec for spec in envs.registry.all()]
@tools.params(*specs)
def test_env(spec):
    skip_mujoco = os.environ.get('TRAVIS_PULL_REQUEST', 'false') != 'false'
    if skip_mujoco and spec.entry_point.startswith('gym.envs.mujoco:'):
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

    for mode in env.metadata.get('render.modes'):
        env.render(mode=mode)

# Run a longer rollout on some environments
def test_random_rollout():
    for env in [envs.make('CartPole-v0'), envs.make('FrozenLake-v0')]:
        agent = lambda ob: env.action_space.sample()
        ob = env.reset()
        for _ in xrange(10):
            assert env.observation_space.contains(ob)
            a = agent(ob)
            assert env.action_space.contains(a)
            (ob, _reward, done, _info) = env.step(a)
            if done: break
