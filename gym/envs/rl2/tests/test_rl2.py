from gym.benchmarks import registration
from gym.monitoring.tests import helpers
from gym.envs import rl2
from gym import monitoring
import gym

import numpy as np


def test_bandits():
    env = rl2.BernoulliBanditEnv(n_arms=5, n_episodes=10)
    env.seed(0)
    obs = env.reset()
    assert isinstance(obs, tuple)
    assert len(obs) == 4
    assert obs[-1] == 1
    # run it twice to make sure behavior is consistent
    for _ in range(2):
        path = random_rollout(env, n_steps=10)
        assert path["dones"][-1]
        assert not np.any(path["dones"][:-1])
        raw_obs, last_actions, last_rewards, last_dones = [list(x) for x in zip(*path["observations"])]
        # test that prev actions and actions taken agree
        np.testing.assert_array_equal(
            last_actions[1:],
            path["actions"][:-1]
        )
        # test that prev rewards and rewards taken agree
        np.testing.assert_array_equal(
            last_rewards[1:],
            path["rewards"][:-1]
        )
        # test that it always terminates
        assert np.all(last_dones)


def test_random_tabular_mdps():
    env = rl2.RandomTabularMDPEnv(n_states=10, n_actions=5, n_episodes=10, episode_length=10)
    env.seed(0)
    obs = env.reset()
    assert isinstance(obs, tuple)
    assert len(obs) == 4
    assert obs[-1] == 1
    # run it twice to make sure behavior is consistent
    for _ in range(2):
        path = random_rollout(env, n_steps=100)
        assert path["dones"][-1]
        assert not np.any(path["dones"][:-1])
        raw_obs, last_actions, last_rewards, last_dones = [list(x) for x in zip(*path["observations"])]
        # test that prev actions and actions taken agree
        np.testing.assert_array_equal(
            last_actions[1:],
            path["actions"][:-1]
        )
        # test that prev rewards and rewards taken agree
        np.testing.assert_array_equal(
            last_rewards[1:],
            path["rewards"][:-1]
        )
        last_dones = np.asarray(last_dones)
        # test that it terminates on the right times
        assert np.all(last_dones[::10])
        last_dones[::10] = 0
        assert not np.any(last_dones)
        # test that it always starts episodes on state 0
        assert np.all(np.asarray(raw_obs[::10]) == 0)


def test_benchmarks():
    for benchmark_id in ['BernoulliBandit-v0', 'RandomTabularMDP-v0']:

        benchmark = registration.benchmark_spec(benchmark_id)

        for env_id in benchmark.env_ids:

            with helpers.tempdir() as temp:
                env = gym.make(env_id)
                env.seed(0)
                env.monitor.start(temp, video_callable=False)

                env.monitor.configure(mode='evaluation')
                rollout(env)

                env.monitor.configure(mode='training')
                for i in range(2):
                    rollout(env)

                env.monitor.configure(mode='evaluation')
                rollout(env, good=True)

                env.monitor.close()
                results = monitoring.load_results(temp)
                evaluation_score = benchmark.score_evaluation(env_id, results['data_sources'],
                                                              results['initial_reset_timestamps'],
                                                              results['episode_lengths'],
                                                              results['episode_rewards'], results['episode_types'],
                                                              results['timestamps'])
                benchmark.score_benchmark({
                    env_id: evaluation_score['scores'],
                })


def rollout(env, good=False):
    env.reset()

    action = 0
    d = False
    while not d:
        if good:
            action = 1 - action
        o, r, d, i = env.step(action)


def random_rollout(env, n_steps):
    observations = []
    actions = []
    rewards = []
    dones = []
    obs = env.reset()
    for _ in range(n_steps):
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        obs = next_obs
    return dict(observations=observations, actions=actions, rewards=rewards, dones=dones)


test_benchmarks()
