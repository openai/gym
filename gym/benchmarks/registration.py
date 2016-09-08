# EXPERIMENTAL: all may be removed soon

import collections
import gym.envs
import logging

from gym import error

logger = logging.getLogger(__name__)

class Task(object):
    def __init__(self, name, env_id, seeds, timesteps):
        self.name = name
        self.env_id = env_id
        self.seeds = seeds
        self.timesteps = timesteps

class Benchmark(object):
    def __init__(self, id, score_method, tasks):
        self.id = id
        self.score_method = score_method
        self.tasks = tasks

class Registry(object):
    def __init__(self):
        self.benchmarks = collections.OrderedDict()

    def add_benchmark(self, id, score_method, tasks):
        task_objects = []
        for task in tasks:
            task_objects.append(Task(
                name=task['name'],
                env_id=task['env_id'],
                seeds=task['seeds'],
                timesteps=task['timesteps'],
            ))

        self.benchmarks[id] = Benchmark(id=id, score_method=score_method, tasks=task_objects)

    def benchmark_spec(self, id):
        try:
            return self.benchmarks[id]
        except KeyError:
            raise error.UnregisteredBenchmark('No registered benchmark with id: {}'.format(id))

registry = Registry()
register_benchmark = registry.register_benchmark
benchmark_spec = registry.benchmark_spec
