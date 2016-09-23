import numpy as np

import gym
from gym import monitoring
from gym.monitoring.tests import helpers

from gym.benchmarks import registration, scoring

def test():
    benchmark = registration.Benchmark(
        id='MyBenchmark-v0',
        scorer=scoring.ClipTo01ThenAverage(),
        task_groups={
            'CartPole-v0': [{
                'seeds': 1,
                'timesteps': 5
            }, {
                'seeds': 1,
                'timesteps': 100
            }],
        })

    with helpers.tempdir() as temp:
        env = gym.make('CartPole-v0')
        env.monitor.start(temp, video_callable=False, seed=0)

        env.monitor.configure(mode='evaluation')
        rollout(env)

        env.monitor.configure(mode='training')
        for i in range(2):
            rollout(env)

        env.monitor.configure(mode='evaluation')
        rollout(env, good=True)

        env.monitor.close()
        results = monitoring.load_results(temp)
        evaluation_score = benchmark.score_evaluation('CartPole-v0', results['episode_lengths'], results['episode_rewards'], results['episode_types'], results['timestamps'], results['initial_reset_timestamp'])
        benchmark_score = benchmark.score_benchmark({
            'CartPole-v0': evaluation_score['scores'],
        })

        assert np.all(np.isclose(evaluation_score['scores'], [0.089999999999999997, 0.27000000000000002])), "evaluation_score={}".format(evaluation_score)
        assert np.isclose(benchmark_score, 0.18), "benchmark_score={}".format(benchmark_score)

def rollout(env, good=False):
    env.reset()

    action = 0
    d = False
    while not d:
        if good:
            action = 1 - action
        o,r,d,i = env.step(action)
