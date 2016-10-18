# EXPERIMENTAL: all may be removed soon

import collections
import gym.envs
import logging

from gym import error

logger = logging.getLogger(__name__)

class Task(object):
    def __init__(self, env_id, seeds, timesteps, reward_floor, reward_ceiling):
        self.env_id = env_id
        self.seeds = seeds
        self.timesteps = timesteps
        self.reward_floor = reward_floor
        self.reward_ceiling = reward_ceiling

class Benchmark(object):
    def __init__(self, id, scorer, task_groups, description=None, name=None):
        self.id = id
        self.scorer = scorer
        self.description = description
        self.name = name

        task_map = {}
        for env_id, tasks in task_groups.items():
            task_map[env_id] = []
            for task in tasks:
                task_map[env_id].append(Task(
                    env_id=env_id,
                    seeds=task['seeds'],
                    timesteps=task['timesteps'],
                    reward_floor=task.get('reward_floor', 0),
                    reward_ceiling=task.get('reward_ceiling', 100),
                ))
        self.task_groups = task_map

    def task_spec(self, env_id):
        try:
            return self.task_groups[env_id]
        except KeyError:
            raise error.Unregistered('No task with env_id {} registered for benchmark {}', env_id, self.id)

    def score_evaluation(self, env_id, episode_lengths, episode_rewards, episode_types, timestamps, initial_reset_timestamp):
        return self.scorer.score_evaluation(self, env_id, episode_lengths, episode_rewards, episode_types, timestamps, initial_reset_timestamp)

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
