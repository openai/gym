import gym
import numpy as np

from gym.benchmarks import ranking
from gym.benchmarks import registration
from gym.scoreboard.tests.test_scoring import _is_close


class MockEvaluation(object):
    def __init__(self, env_id, mean_auc):
        self.env_id = env_id
        self.mean_auc = mean_auc


class MockBenchmarkScoreCache(object):
    def __init__(self, benchmark_id, min_reward_by_env, max_reward_by_env):
        self._min_reward_by_env = min_reward_by_env
        self._max_reward_by_env = max_reward_by_env

        self.id = benchmark_id

    def min_score(self, task_spec):
        """The worst performance we've seen on this task in this benchmark"""
        try:
            return self._min_reward_by_env[task_spec.env_id]
        except KeyError:
            return None

    def max_score(self, task_spec):
        """The best performance we've seen on this task in this benchmark"""
        try:
            return self._max_reward_by_env[task_spec.env_id]
        except KeyError:
            return None

    @property
    def spec(self):
        return gym.benchmark_spec(self.id)


# Create a benchmark
benchmark = registration.Benchmark(
    id='TestSingleTask-v0',
    scorer=None,
    tasks=[
        {
            'env_id': 'TestEnv-v0',
            'trials': 2,
            'max_timesteps': 5
        }]
)

score_cache = MockBenchmarkScoreCache('TestSingleTask-v0',
    min_reward_by_env={
        'TestEnv-v0': 0.0,
    },
    max_reward_by_env={
        'TestEnv-v0': 100.0,
    }
)

test_task_spec = benchmark.task_specs('TestEnv-v0')[0]


def test_task_rank():
    evaluations = [
        MockEvaluation('TestEnv-v0', mean_auc=10.),
        MockEvaluation('TestEnv-v0', mean_auc=20.),
    ]

    task_rank = ranking.compute_task_rank(test_task_spec, score_cache, evaluations)
    assert task_rank == 0.15


def test_task_rank_incomplete_task():
    evaluations = [
        MockEvaluation('TestEnv-v0', mean_auc=20.),
    ]
    task_rank = ranking.compute_task_rank(test_task_spec, score_cache, evaluations)
    assert task_rank == 0.10, "Missing tasks should default to the lowest score we've seen"


def test_task_rank_no_tasks():
    evaluations = []
    task_rank = ranking.compute_task_rank(test_task_spec, score_cache, evaluations)
    assert task_rank == 0., "Missing evaluations should default to the lowest score we've seen"


def test_task_rank_too_many_evaluations():
    evaluations = [
        MockEvaluation('TestEnv-v0', mean_auc=10.),
        MockEvaluation('TestEnv-v0', mean_auc=20.),
        MockEvaluation('TestEnv-v0', mean_auc=30.),
    ]
    task_rank = ranking.compute_task_rank(test_task_spec, score_cache, evaluations)
    assert task_rank == 0.20, "If there are too many evaluations, use the mean of all of them"


def test_compute_benchmark_run_rank():
    evaluations = [
        MockEvaluation('TestEnv-v0', mean_auc=10.),
        MockEvaluation('TestEnv-v0', mean_auc=20.),
    ]

    benchmark_run_rank = ranking.compute_benchmark_run_rank(
        benchmark=benchmark,
        score_cache=score_cache,
        evaluations=evaluations)
    assert benchmark_run_rank == 0.15



multiple_task_benchmark = registration.Benchmark(
    id='TestMultipleTasks-v0',
    scorer=None,
    tasks=[
        {
            'env_id': 'TestEnv-v0',
            'trials': 2,
            'max_timesteps': 5
        },
        {
            'env_id': 'AnotherTestEnv-v0',
            'trials': 1,
            'max_timesteps': 100,
        }]
)

multiple_task_score_cache = MockBenchmarkScoreCache('TestMultipleTasks-v0',
    min_reward_by_env={
        'TestEnv-v0': 0.0,
        'AnotherTestEnv-v0': 0.0,
    },
    max_reward_by_env={
        'TestEnv-v0': 100.0,
        'AnotherTestEnv-v0': 100.0,
    }
)


def test_compute_benchmark_run_rank_multiple_tasks():
    evaluations = [
        MockEvaluation('TestEnv-v0', mean_auc=10.),
        MockEvaluation('TestEnv-v0', mean_auc=10.),
        MockEvaluation('AnotherTestEnv-v0', mean_auc=20.),
    ]

    benchmark_run_rank = ranking.compute_benchmark_run_rank(
        benchmark=multiple_task_benchmark,
        score_cache=multiple_task_score_cache,
        evaluations=evaluations)
    assert _is_close(benchmark_run_rank, 0.15)
