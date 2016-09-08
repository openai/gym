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

    def task_spec(self, env_id):
        for task in self.tasks:
            if task.env_id == env_id:
                return task
        raise error.Unregistered('No task with env_id {} registered for benchmark {}', env_id, self.id)

class Registry(object):
    def __init__(self):
        self.benchmarks = collections.OrderedDict()

    def register_benchmark(self, id, score_method, tasks):
        task_objects = []
        for name, task in tasks.items():
            task_objects.append(Task(
                name=name,
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
