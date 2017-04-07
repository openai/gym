#!/usr/bin/env python3
import io
import logging
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from flask import Flask
from flask import render_template
from scipy import signal

from gym import monitoring
from gym.benchmarks import registry

app = Flask(__name__)

try:
    BENCHMARK_VIEWER_DATA_PATH = os.environ['BENCHMARK_VIEWER_DATA_PATH'].rstrip('/')
except KeyError:
    print(
        "Missing BENCHMARK_VIEWER_DATA_PATH environment variable. Run the viewer with `BENCHMARK_VIEWER_DATA_PATH=/tmp/AtariExploration40M ./app.py` ")
    sys.exit()

BENCHMARK_ID = os.path.basename(BENCHMARK_VIEWER_DATA_PATH)

logger = logging.getLogger(__name__)


class BenchmarkResource(object):
    def __init__(self, id, data_path, bmruns):
        self.id = id
        self.data_path = data_path
        self.bmruns = bmruns


class BenchmarkCache(object):
    """
    Stores data about the benchmark in memory
    """

    def __init__(self, benchmark_id):
        self.min_reward_by_env = {}
        self.max_reward_by_env = {}
        self.bmruns = []

        self.id = benchmark_id

    def benchmark_resource(self):
        return BenchmarkResource(
            id=self.id,
            data_path=BENCHMARK_VIEWER_DATA_PATH,
            bmruns=self.bmruns
        )


# singleton benchmark_cache
benchmark_cache = BenchmarkCache(BENCHMARK_ID)


class EvaluationResource(object):
    def __init__(self, env_id, results):
        self.env_id = env_id

        self.episode_rewards = results['episode_rewards']
        self.episode_lengths = results['episode_lengths']
        self.episode_types = results['episode_types']
        self.timestamps = results['timestamps']
        self.initial_reset_timestamps = results['initial_reset_timestamps']
        self.data_sources = results['data_sources']

    @property
    def score(self):
        benchmark = registry.benchmark_spec(BENCHMARK_ID)

        score_results = benchmark.score_evaluation(
            self.env_id,
            data_sources=self.data_sources,
            initial_reset_timestamps=self.initial_reset_timestamps,
            episode_lengths=self.episode_lengths,
            episode_rewards=self.episode_rewards,
            episode_types=self.episode_types,
            timestamps=self.timestamps)

        # TODO: Why does the scorer output vectorized here?
        return mean_area_under_curve(
            score_results['lengths'][0],
            score_results['rewards'][0],
        )


def load_evaluations_from_bmrun_path(path):
    evaluations = []
    for training_dir in glob('{}/*/gym'.format(path)):

        results = monitoring.load_results(training_dir)
        if not results:
            logger.info("Failed to load data for %s" % training_dir)
        else:
            evaluation = EvaluationResource(results['env_info']['env_id'], results)
            evaluations.append(evaluation)

    return evaluations


def load_tasks_from_bmrun_path(path):
    env_id_to_task = {}

    for evaluation in load_evaluations_from_bmrun_path(path):

        env_id = evaluation.env_id

        if env_id not in env_id_to_task:
            env_id_to_task[env_id] = Task(env_id, benchmark_id=BENCHMARK_ID, evaluations=[])
        task = env_id_to_task[env_id]

        task.evaluations.append(evaluation)

    return env_id_to_task.values()


class BenchmarkRunResource(object):
    def __init__(self, path, tasks):
        self.tasks = sorted(tasks, key=lambda t: t.env_id)
        self.name = os.path.basename(path)
        self.path = path

    @property
    def short_name(self):
        return '_'.join(self.name.split('_')[2:])

    @classmethod
    def from_path(cls, bmrun_path):
        tasks = load_tasks_from_bmrun_path(bmrun_path)
        return cls(bmrun_path, tasks)


def smooth_reward_curve(rewards, lengths, max_timestep, resolution=1e3, polyorder=3):
    # Don't use a higher resolution than the original data, use a window about
    # 1/10th the resolution
    resolution = min(len(rewards), resolution)
    window = int(resolution / 10)
    window = window + 1 if (window % 2 == 0) else window  # window must be odd

    if polyorder >= window:
        return lengths, rewards

    # Linear interpolation, followed by Savitzky-Golay filter
    x = np.cumsum(np.array(lengths, 'float'))
    y = np.array(rewards, 'float')

    x_spaced = np.linspace(0, max_timestep, resolution)
    y_interpolated = np.interp(x_spaced, x, y)
    y_smoothed = signal.savgol_filter(y_interpolated, window, polyorder=polyorder)

    return x_spaced.tolist(), y_smoothed.tolist()


class Task(object):
    def __init__(self, env_id, benchmark_id, evaluations):
        self.env_id = env_id
        self.benchmark_id = benchmark_id
        self.evaluations = evaluations

    @property
    def score(self):
        return np.mean([eval.score for eval in self.evaluations])

    @property
    def spec(self):
        benchmark_spec = registry.benchmark_spec(self.benchmark_id)
        task_specs = benchmark_spec.task_specs(self.env_id)
        if len(task_specs) != 1:
            raise Error("Multiple task specs for single environment. Falling over")

        return task_specs[0]

    def render_learning_curve_svg(self):
        plt.figure()
        plt.rcParams['figure.figsize'] = (8, 2)

        for trial in self.evaluations:
            xs, ys = smooth_reward_curve(
                trial.episode_rewards, trial.episode_lengths, self.spec.max_timesteps)
            plt.plot(xs, ys)

        plt.xlabel('Time')
        plt.ylabel('Rewards')
        plt.tight_layout()
        img_bytes = io.StringIO()
        plt.savefig(img_bytes, format='svg')
        return img_bytes.getvalue()


def area_under_curve(episode_lengths, episode_rewards):
    """Compute the total area of rewards under the curve"""
    # TODO: Replace with slightly more accurate trapezoid method
    return np.sum(l * r for l, r in zip(episode_lengths, episode_rewards))


def mean_area_under_curve(episode_lengths, episode_rewards):
    """Compute the average area of rewards under the curve per unit of time"""
    return area_under_curve(episode_lengths, episode_rewards) / max(1e-4, np.sum(episode_lengths))


def _benchmark_runs_from_dir(benchmark_dir):
    run_paths = [os.path.join(benchmark_dir, path) for path in os.listdir(benchmark_dir)]
    run_paths = [path for path in run_paths if os.path.isdir(path)]
    return [BenchmarkRunResource.from_path(path) for path in run_paths]


@app.route('/')
def index():
    return render_template('benchmark.html', benchmark=benchmark_cache.benchmark_resource())


@app.route('/compare/<run_name>/<other_run_name>/')
def compare(run_name, other_run_name):
    pass


@app.route('/benchmark_run/<bmrun_name>')
def benchmark_run(bmrun_name):
    bmrun_dir = os.path.join(BENCHMARK_VIEWER_DATA_PATH, bmrun_name)
    bmrun = BenchmarkRunResource.from_path(bmrun_dir)
    #
    # rows = ''.join(
    #     '<tr><td>{}</td><td>{}</td></tr>'.format(env_id, task.to_svg())
    #         for env_id, task in sorted(tasks.items())
    # )
    # return '<table>{}</tbody>'.format(rows)

    return render_template('benchmark_run.html', bmrun=bmrun)


def populate_benchmark_cache():
    bmruns = _benchmark_runs_from_dir(BENCHMARK_VIEWER_DATA_PATH)

    benchmark_cache.bmruns = bmruns


if __name__ == '__main__':
    populate_benchmark_cache()
    app.run(debug=True, port=5000)
