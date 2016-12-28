import numpy as np

import gym
from gym import monitoring, wrappers
from gym.monitoring.tests import helpers

from gym.benchmarks import registration, scoring

def test():
    benchmark = registration.Benchmark(
        id='MyBenchmark-v0',
        scorer=scoring.ClipTo01ThenAverage(),
        tasks=[
            {'env_id': 'CartPole-v0',
             'trials': 1,
             'max_timesteps': 5
            },
            {'env_id': 'CartPole-v0',
             'trials': 1,
             'max_timesteps': 100,
            }])

    with helpers.tempdir() as temp:
        env = gym.make('CartPole-v0')
        env = wrappers.Monitor(env, directory=temp, video_callable=False)
        env.seed(0)

        env.set_monitor_mode('evaluation')
        rollout(env)

        env.set_monitor_mode('training')
        for i in range(2):
            rollout(env)

        env.set_monitor_mode('evaluation')
        rollout(env, good=True)

        env.close()
        results = monitoring.load_results(temp)
        evaluation_score = benchmark.score_evaluation('CartPole-v0', results['data_sources'], results['initial_reset_timestamps'], results['episode_lengths'], results['episode_rewards'], results['episode_types'], results['timestamps'])
        benchmark_score = benchmark.score_benchmark({
            'CartPole-v0': evaluation_score['scores'],
        })

        assert np.all(np.isclose(evaluation_score['scores'], [0.00089999999999999998, 0.0054000000000000003])), "evaluation_score={}".format(evaluation_score)
        assert np.isclose(benchmark_score, 0.00315), "benchmark_score={}".format(benchmark_score)

def rollout(env, good=False):
    env.reset()

    action = 0
    d = False
    while not d:
        if good:
            action = 1 - action
        o,r,d,i = env.step(action)
