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
        self.env_ids = set()

        compiled_tasks = []
        for task in tasks:
            task = Task(
                env_id=task['env_id'],
                trials=task['trials'],
                max_timesteps=task.get('max_timesteps'),
                max_seconds=task.get('max_seconds'),
                reward_floor=task.get('reward_floor', 0),
                reward_ceiling=task.get('reward_ceiling', 100),
            )
            self.env_ids.add(task.env_id)
            compiled_tasks.append(task)

        self.tasks = compiled_tasks

    def task_specs(self, env_id):
        # Could precompute this, but no need yet
        # Note that if we do precompute it we need to preserve the order in
        # which tasks are returned
        results = [task for task in self.tasks if task.env_id == env_id]
        if not results:
            raise error.Unregistered('No task with env_id {} registered for benchmark {}', env_id, self.id)
        return results

    def score_evaluation(self, env_id, data_sources, initial_reset_timestamps, episode_lengths, episode_rewards, episode_types, timestamps):
        return self.scorer.score_evaluation(self, env_id, data_sources, initial_reset_timestamps, episode_lengths, episode_rewards, episode_types, timestamps)

    def score_benchmark(self, score_map):
        return self.scorer.score_benchmark(self, score_map)

BenchmarkView = collections.namedtuple("BenchmarkView", ["name", "benchmarks", "primary", "group"])

class Registry(object):
    def __init__(self):
        self.benchmarks            = collections.OrderedDict()
        self.benchmark_views       = collections.OrderedDict()
        self.benchmark_view_groups = collections.OrderedDict()

    def register_benchmark_view(self, name, benchmarks, primary, group):
        """Sometimes there's very little change between one
        benchmark and another. BenchmarkView will allow to
        display results from multiple benchmarks in a single
        table.

        name: str
            Name to display on the website
        benchmarks: [str]
            list of benchmark ids to include
        primary: str
            primary benchmark - this is one to be used
            to display as the most recent benchmark to be
            used when submitting for future evaluations.
        group: str
            group in which to display the benchmark on the website.
        """
        assert name.replace("_", '').replace('-', '').isalnum(), \
                "Name of benchmark must be combination of letters, numbers, - and _"
        if group is None:
            group = "Miscellaneous"
        bw = BenchmarkView(name=name, benchmarks=benchmarks, primary=primary, group=group)
        assert bw.primary in bw.benchmarks
        self.benchmark_views[bw.name] = bw
        if group not in self.benchmark_view_groups:
            self.benchmark_view_groups[group] = []
        self.benchmark_view_groups[group].append(bw)

    def register_benchmark(self, id, scorer, tasks, description=None, name=None, add_view=True, view_group=None):
        self.benchmarks[id] = Benchmark(id=id, scorer=scorer, tasks=tasks, name=name, description=description)
        if add_view:
            self.register_benchmark_view(name=name if name is not None else id,
                                         benchmarks=[id],
                                         primary=id,
                                         group=view_group)

    def benchmark_spec(self, id):
        try:
            return self.benchmarks[id]
        except KeyError:
            raise error.UnregisteredBenchmark('No registered benchmark with id: {}'.format(id))

registry = Registry()
register_benchmark      = registry.register_benchmark
register_benchmark_view = registry.register_benchmark_view
benchmark_spec          = registry.benchmark_spec
