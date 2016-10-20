# EXPERIMENTAL: all may be removed soon

import collections
import gym.envs
import logging

from gym import error

logger = logging.getLogger(__name__)

class Task(object):
    def __init__(self, env_id, trials, max_timesteps, max_seconds, reward_floor, reward_ceiling):
        self.env_id = env_id
        self.trials = trials
        self.max_timesteps = max_timesteps
        self.max_seconds = max_seconds
        self.reward_floor = reward_floor
        self.reward_ceiling = reward_ceiling

        if max_timesteps is None and max_seconds is None:
            raise error.Error('Must provide at least one of max_timesteps and max_seconds for {}'.format(self))

    def __str__(self):
        return 'Task<env_id={} trials={} max_timesteps={} max_seconds={} reward_floor={} reward_ceiling={}>'.format(self.env_id, self.trials, self.max_timesteps, self.max_seconds, self.reward_floor, self.reward_ceiling)

class Benchmark(object):
    def __init__(self, id, scorer, tasks, description=None, name=None):
        self.id = id
        self.scorer = scorer
        self.description = description
        self.name = name

        compiled_tasks = []
        for task in tasks:
            compiled_tasks.append(Task(
                env_id=task['env_id'],
                trials=task['trials'],
                max_timesteps=task.get('max_timesteps'),
                max_seconds=task.get('max_seconds'),
                reward_floor=task.get('reward_floor', 0),
                reward_ceiling=task.get('reward_ceiling', 100),
            ))
        self.tasks = compiled_tasks

    def task_specs(self, env_id):
        try:
            # Could precompute this, but no need yet
            return [task for task in self.tasks if task.env_id == env_id]
        except KeyError:
            raise error.Unregistered('No task with env_id {} registered for benchmark {}', env_id, self.id)

    def score_evaluation(self, env_id, data_sources, initial_reset_timestamps, episode_lengths, episode_rewards, episode_types, timestamps):
        return self.scorer.score_evaluation(self, env_id, data_sources, initial_reset_timestamps, episode_lengths, episode_rewards, episode_types, timestamps)

    def score_benchmark(self, score_map):
        return self.scorer.score_benchmark(self, score_map)

class Registry(object):
    def __init__(self):
        self.benchmarks = collections.OrderedDict()

    def register_benchmark(self, id, **kwargs):
        self.benchmarks[id] = Benchmark(id=id, **kwargs)

    def benchmark_spec(self, id):
        try:
            return self.benchmarks[id]
        except KeyError:
            raise error.UnregisteredBenchmark('No registered benchmark with id: {}'.format(id))

registry = Registry()
register_benchmark = registry.register_benchmark
benchmark_spec = registry.benchmark_spec
