import numpy as np
from collections import defaultdict
from gym.benchmarks import registration, scoring
from gym.scoreboard.scoring import benchmark_aggregate_results

import gym
gym.undo_logger_setup()

benchmark = registration.Benchmark(
    id='TestBenchmark-v0',
    scorer=scoring.ClipTo01ThenAverage(),
    tasks=[
        {'env_id': 'CartPole-v0',
         'trials': 1,
         'max_timesteps': 5,
        },
        {'env_id': 'Pendulum-v0',
         'trials': 1,
         'max_timesteps': 5,
        },
    ]
)

def _is_close(x, target):
    return np.all(np.isclose(x, target))

def _benchmark_result_helper(benchmark, **kwargs):
    for k, defval in dict(
            env_id='CartPole-v0',
            data_sources=[0],
            initial_reset_timestamps=[1],
            episode_lengths=[1],
            episode_rewards=[1],
            episode_types=['t'],
            timestamps=[2]).items():
        kwargs.setdefault(k, defval)

    return benchmark.score_evaluation(**kwargs)

def test_clip_scoring():
    benchmark = registration.Benchmark(
        id='TestBenchmark-v0',
        scorer=scoring.ClipTo01ThenAverage(num_episodes=1),
        tasks=[
            {'env_id': 'CartPole-v0',
             'trials': 1,
             'max_timesteps': 5,
            },
        ]
    )
    # simple scoring
    benchmark_result = _benchmark_result_helper(benchmark)
    assert _is_close(benchmark_result['scores'][0], 0.01), "benchmark_result={}".format(benchmark_result)

    # test a successful run
    benchmark_result = _benchmark_result_helper(benchmark, episode_rewards=[100])
    assert _is_close(benchmark_result['scores'][0], 1.0), "benchmark_result={}".format(benchmark_result)
    assert np.all(benchmark_result['solves'][0]), "benchmark_result={}".format(benchmark_result)

def test_max_timesteps():
    benchmark = registration.Benchmark(
        id='TestBenchmark-v0',
        scorer=scoring.ClipTo01ThenAverage(num_episodes=2),
        tasks=[
            {'env_id': 'CartPole-v0',
             'trials': 1,
             'max_timesteps': 2,
            },
        ]
    )

    benchmark_result = _benchmark_result_helper(benchmark, data_sources=[0,0], episode_lengths=[1,1], episode_rewards=[1,1], episode_types=['t','t'], timestamps=[2,3])
    assert _is_close(benchmark_result['scores'][0], 0.01), "benchmark_result={}".format(benchmark_result)

    # make sure we only include the first result because of timesteps
    benchmark_result = _benchmark_result_helper(benchmark, data_sources=[0,0], episode_lengths=[1,100], episode_rewards=[1,100], episode_types=['t','t'], timestamps=[2,102])
    assert _is_close(benchmark_result['scores'][0], 0.005), "benchmark_result={}".format(benchmark_result)
    assert not np.any(benchmark_result['solves'][0]), "benchmark_result={}".format(benchmark_result)

def test_max_seconds():
    benchmark = registration.Benchmark(
        id='TestBenchmark-v0',
        scorer=scoring.ClipTo01ThenAverage(num_episodes=2),
        tasks=[
            {'env_id': 'CartPole-v0',
             'trials': 1,
             'max_seconds': 1,
            },
        ]
    )

    benchmark_result = _benchmark_result_helper(benchmark, data_sources=[0,0], episode_lengths=[100,100], episode_rewards=[0,100], episode_types=['t','t'], timestamps=[1.5, 2])
    assert _is_close(benchmark_result['scores'][0], 0.5), "benchmark_result={}".format(benchmark_result)

    # make sure we only include the first result because of wall clock time
    benchmark_result = _benchmark_result_helper(benchmark, data_sources=[0,0], episode_lengths=[100,100], episode_rewards=[0,100], episode_types=['t','t'], timestamps=[2,100])
    assert _is_close(benchmark_result['scores'][0], 0.0), "benchmark_result={}".format(benchmark_result)

def test_benchmark_scoring():
    benchmark_results = defaultdict(list)
    for i, task in enumerate(benchmark.tasks):
        env_id = task.env_id
        benchmark_results[env_id].append(benchmark.score_evaluation(
            env_id,
            data_sources=[0],
            initial_reset_timestamps=[1],
            episode_lengths=[1],
            episode_rewards=[1],
            episode_types=['t'],
            timestamps=[i + 2],
        ))
    scores = benchmark_aggregate_results(benchmark, benchmark_results)

    debug_str = "scores={}".format(scores)
    assert np.all(np.isclose(scores['summed_training_seconds'], 3.0)), debug_str
    assert np.all(np.isclose(scores['start_to_finish_seconds'], 2.0)), debug_str
    assert np.all(np.isclose(scores['score'], 0.0001)), "scores={}".format(scores)
    assert scores['num_envs_solved'] == 0, debug_str

def test_benchmark_solved():
    benchmark_results = defaultdict(list)
    N = 200
    for i, task in enumerate(benchmark.tasks):
        env_id = task.env_id
        benchmark_results[env_id].append(benchmark.score_evaluation(
            env_id,
            data_sources=[0],
            initial_reset_timestamps=[1],
            episode_lengths=[1000] * N,
            episode_rewards=[1000] * N,
            episode_types=['t'] * N,
            timestamps=list(range(N)),
        ))
    scores = benchmark_aggregate_results(benchmark, benchmark_results)
    debug_str = "scores={}".format(scores)
    assert np.all(np.isclose(scores['score'], 1.0)), "scores={}".format(scores)
    assert scores['num_envs_solved'] == len(benchmark.tasks), debug_str

def test_benchmark_incomplete():
    benchmark_results = defaultdict(list)
    env_id = benchmark.tasks[0].env_id
    benchmark_results[env_id].append(benchmark.score_evaluation(
        env_id,
        data_sources=[0],
        initial_reset_timestamps=[1],
        episode_lengths=[1],
        episode_rewards=[1],
        episode_types=['t'],
        timestamps=[2],
    ))
    scores = benchmark_aggregate_results(benchmark, benchmark_results)

    debug_str = "scores={}".format(scores)
    assert np.all(np.isclose(scores['summed_training_seconds'], 1.0)), debug_str
    assert np.all(np.isclose(scores['start_to_finish_seconds'], 1.0)), debug_str
    assert np.all(np.isclose(scores['score'], 0.00005)), "scores={}".format(scores)
    assert scores['num_envs_solved'] == 0, debug_str

def test_benchmark_extra():
    benchmark_results = defaultdict(list)
    for i, task in enumerate(benchmark.tasks):
        env_id = task.env_id
        benchmark_results[env_id].append(benchmark.score_evaluation(
            env_id,
            data_sources=[0],
            initial_reset_timestamps=[1],
            episode_lengths=[1],
            episode_rewards=[1],
            episode_types=['t'],
            timestamps=[i + 2],
        ))

    # add one more at the end with a high reward
    benchmark_results[env_id].append(benchmark.score_evaluation(
        env_id,
        data_sources=[0],
        initial_reset_timestamps=[1],
        episode_lengths=[1],
        episode_rewards=[100],
        episode_types=['t'],
        timestamps=[2],
    ))

    scores = benchmark_aggregate_results(benchmark, benchmark_results)

    debug_str = "scores={}".format(scores)
    assert np.all(np.isclose(scores['score'], 0.0001)), "scores={}".format(scores)
    assert np.all(np.isclose(scores['summed_training_seconds'], 3.0)), debug_str
    assert np.all(np.isclose(scores['start_to_finish_seconds'], 2.0)), debug_str
    assert scores['num_envs_solved'] == 0, debug_str
