#!/usr/bin/env python3
import argparse
import io
import logging
import os
import subprocess
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from flask import Flask
from flask import render_template
from gym import monitoring
from gym.benchmarks import registry
from gym.benchmarks.viewer.error import Error
from scipy import signal

logger = logging.getLogger(__name__)

#############################
# Args
#############################

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',
    help="The path to our benchmark data. e.g. /tmp/Atari40M/ ")
parser.add_argument('--debug', action="store_true",
    help="Run with debugger and auto-reload")
parser.add_argument('--flush-cache', action="store_true",
    help="NOT YET IMPLEMENTED: Flush the cache and recompute ranks from scratch")
parser.add_argument('--watch', action="store_true",
    help="NOT YET IMPLEMENTED: Watch for changes and refresh")
parser.add_argument('--open', action="store_true",
    help="Open the browser")
ARGS = parser.parse_args()

BENCHMARK_VIEWER_DATA_PATH = ARGS.data_path.rstrip('/')

BENCHMARK_ID = os.path.basename(BENCHMARK_VIEWER_DATA_PATH)

app = Flask(__name__)


#############################
# Cache
#############################

class BenchmarkCache(object):
    """
    Stores data about the benchmark in memory
    """

    def __init__(self, benchmark_id):
        self.min_reward_by_env = {}
        self.max_reward_by_env = {}
        self.bmruns = []

        self.id = benchmark_id


# singleton benchmark_cache
benchmark_cache = BenchmarkCache(BENCHMARK_ID)


#############################
# Resources
#############################

class BenchmarkResource(object):
    def __init__(self, id, data_path, bmruns):
        self.id = id
        self.data_path = data_path
        self.bmruns = bmruns


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


class TaskResource(object):
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
        return render_evaluation_learning_curves_svg(self.evaluations, self.spec.max_timesteps)


#############################
# Graph rendering
#############################

def area_under_curve(episode_lengths, episode_rewards):
    """Compute the total area of rewards under the curve"""
    # TODO: Replace with slightly more accurate trapezoid method
    return np.sum(l * r for l, r in zip(episode_lengths, episode_rewards))


def mean_area_under_curve(episode_lengths, episode_rewards):
    """Compute the average area of rewards under the curve per unit of time"""
    return area_under_curve(episode_lengths, episode_rewards) / max(1e-4, np.sum(episode_lengths))


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


def render_evaluation_learning_curves_svg(evaluations, max_timesteps):
    plt.figure()
    plt.rcParams['figure.figsize'] = (8, 2)

    for evaluation in evaluations:
        xs, ys = smooth_reward_curve(
            evaluation.episode_rewards, evaluation.episode_lengths, max_timesteps)
        plt.plot(xs, ys)

    plt.xlabel('Time')
    plt.ylabel('Rewards')
    plt.tight_layout()
    img_bytes = io.StringIO()
    plt.savefig(img_bytes, format='svg')
    return img_bytes.getvalue()


#############################
# Controllers
#############################

@app.route('/')
def index():
    benchmark = BenchmarkResource(
        id=benchmark_cache.id,
        data_path=BENCHMARK_VIEWER_DATA_PATH,
        bmruns=benchmark_cache.bmruns
    )

    return render_template('benchmark.html', benchmark=benchmark)


@app.route('/compare/<run_name>/<other_run_name>/')
def compare(run_name, other_run_name):
    pass


@app.route('/benchmark_run/<bmrun_name>')
def benchmark_run(bmrun_name):
    bmrun_dir = os.path.join(BENCHMARK_VIEWER_DATA_PATH, bmrun_name)
    bmrun = BenchmarkRunResource.from_path(bmrun_dir)

    # Hack that warms up pyplot. Renders and drops result on floor
    # TODO: Fix pyplot
    if bmrun.tasks[0]:
        bmrun.tasks[0].render_learning_curve_svg()

    return render_template('benchmark_run.html', bmrun=bmrun)


#############################
# Data loading
#############################

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
            env_id_to_task[env_id] = TaskResource(env_id, benchmark_id=BENCHMARK_ID, evaluations=[])
        task = env_id_to_task[env_id]

        task.evaluations.append(evaluation)

    return env_id_to_task.values()


def _benchmark_runs_from_dir(benchmark_dir):
    run_paths = [os.path.join(benchmark_dir, path) for path in os.listdir(benchmark_dir)]
    run_paths = [path for path in run_paths if os.path.isdir(path)]
    return [BenchmarkRunResource.from_path(path) for path in run_paths]


def populate_benchmark_cache():
    bmruns = _benchmark_runs_from_dir(BENCHMARK_VIEWER_DATA_PATH)
    benchmark_cache.bmruns = bmruns

    # Populate min and max tasks
    for run in bmruns:
        pass


if __name__ == '__main__':
    populate_benchmark_cache()

    if ARGS.open:
        subprocess.check_call('open http://localhost:5000', shell=True)

    app.run(debug=ARGS.debug, port=5000)
