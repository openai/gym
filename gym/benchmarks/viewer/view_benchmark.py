#!/usr/bin/env python3
import argparse
import io
import logging
import os
import subprocess
from glob import glob

import gym
import matplotlib.pyplot as plt
import numpy as np
import time

import yaml
from flask import Flask
from flask import render_template
from gym import monitoring
from gym.benchmarks import registry
from gym.benchmarks.viewer.error import Error
from gym.benchmarks.viewer.template_helpers import register_template_helpers
from scipy import signal

logger = logging.getLogger(__name__)

#############################
# Args
#############################

parser = argparse.ArgumentParser()
parser.add_argument('data_path',
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
BENCHMARK_SPEC = gym.benchmark_spec(BENCHMARK_ID)

app = Flask(__name__)


#############################
# Cache
#############################

class ScoreCache(object):
    """
    Stores data about the benchmark in memory
    """

    def __init__(self, benchmark_id):
        self._env_id_to_min_scoring_bmrun = {}
        self._env_id_to_max_scoring_bmrun = {}

        # Maps evaluation_dir to score, and date added
        self._evaluation_cache = {}

        self.id = benchmark_id

    def cache_evaluation_score(self, bmrun, evaluation, score):
        env_id = evaluation.env_id

        # See if we should overwrite the min or max
        if self.min_score(env_id) is None or score < self.min_score(env_id):
            self._env_id_to_min_scoring_bmrun[env_id] = {
                'bmrun_name': bmrun.name,
                'score': score
            }

        if self.max_score(env_id) is None or score > self.max_score(env_id):
            self._env_id_to_max_scoring_bmrun[env_id] = {
                'bmrun_name': bmrun.name,
                'score': score
            }

        # Cache the time also, so we know when to cachebust
        self._evaluation_cache[evaluation] = {
            'score': score,
            'score_cached_at': time.time()
        }

    def min_score(self, env_id):
        """The worst evaluation performance we've seen on this env on this benchmark"""
        try:
            return self._env_id_to_min_scoring_bmrun[env_id]['score']
        except KeyError:
            return None

    def max_score(self, env_id):
        """The best evaluation performance we've seen on this env on this benchmark"""
        try:
            return self._env_id_to_max_scoring_bmrun[env_id]['score']
        except KeyError:
            return None

    @property
    def spec(self):
        return gym.benchmark_spec(self.id)


score_cache = ScoreCache(BENCHMARK_ID)


#############################
# Resources
#############################

class BenchmarkResource(object):
    def __init__(self, id, data_path, bmruns):
        self.id = id
        self.data_path = data_path
        self.bmruns = bmruns

    @property
    def spec(self):
        return gym.benchmark_spec(self.id)


class EvaluationResource(object):
    def __init__(self, env_id, results, evaluation_dir):
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
    def __init__(self, path, tasks, metadata=None):
        self.tasks = sorted(tasks, key=lambda t: t.env_id)
        self.name = os.path.basename(path)
        self.path = path

        if metadata:
            self.author_username = metadata['author']['username']
            self.author_github = metadata['author']['github_user']
            self.repository = metadata['code_source']['repository']
            self.github_commit = metadata['code_source']['commit']
        else:
            self.author_username = None
            self.author_github = None
            self.repository = None
            self.github_commit = None

    @property
    def short_name(self):
        return '_'.join(self.name.split('_')[2:])

    def task_by_env_id(self, env_id):
        return [task for task in self.tasks if task.env_id == env_id][0]

    @property
    def evaluations(self):
        return [evaluation for task in self.tasks for evaluation in task.evaluations]


class TaskResource(object):
    def __init__(self, env_id, benchmark_id, evaluations):
        self.env_id = env_id
        self.benchmark_id = benchmark_id
        self.evaluations = evaluations

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
        id=BENCHMARK_ID,
        data_path=BENCHMARK_VIEWER_DATA_PATH,
        bmruns=_benchmark_runs_from_dir(BENCHMARK_VIEWER_DATA_PATH)
    )

    return render_template('benchmark.html',
        benchmark=benchmark,
        score_cache=score_cache
    )


@app.route('/compare/<run_name>/<other_run_name>/')
def compare(run_name, other_run_name):
    pass


@app.route('/benchmark_run/<bmrun_name>')
def benchmark_run(bmrun_name):
    bmrun_dir = os.path.join(BENCHMARK_VIEWER_DATA_PATH, bmrun_name)
    bmrun = load_bmrun_from_path(bmrun_dir)

    # Hack that warms up pyplot. Renders and drops result on floor
    # TODO: Fix pyplot
    if bmrun.tasks[0]:
        bmrun.tasks[0].render_learning_curve_svg()

    return render_template('benchmark_run.html',
        bmrun=bmrun,
        benchmark_spec=BENCHMARK_SPEC,
        score_cache=score_cache
    )


#############################
# Data loading
#############################



def load_evaluations_from_bmrun_path(path):
    evaluations = []

    dirs_missing_manifests = []
    dirs_unloadable = []

    for training_dir in glob('%s/*/gym' % path):

        if not monitoring.detect_training_manifests(training_dir):
            dirs_missing_manifests.append(training_dir)
            continue
        results = monitoring.load_results(training_dir)
        if not results:
            dirs_unloadable.append(training_dir)
            continue
        else:
            evaluation = EvaluationResource(results['env_info']['env_id'], results, training_dir)
            evaluations.append(evaluation)

    if dirs_missing_manifests:
        logger.warning("Could not load %s evaluations in %s due to missing manifests" % (len(dirs_missing_manifests), path))

    if dirs_unloadable:
        logger.warning("monitoring.load_results failed on %s evaluations in %s" % (len(dirs_unloadable), path))

    return evaluations


def load_tasks_from_bmrun_path(path):
    env_id_to_task = {}
    for task in BENCHMARK_SPEC.tasks:
        env_id_to_task[task.env_id] = TaskResource(task.env_id, benchmark_id=BENCHMARK_ID,
            evaluations=[])

    for evaluation in load_evaluations_from_bmrun_path(path):
        env_id = evaluation.env_id
        env_id_to_task[env_id].evaluations.append(evaluation)

    return env_id_to_task.values()


def load_bmrun_from_path(path):
    tasks = load_tasks_from_bmrun_path(path)

    # Load in metadata from yaml
    metadata = None
    yaml_file = os.path.join(path, 'benchmark_run_data.yaml')
    if os.path.isfile(yaml_file):
        with open(yaml_file, 'r') as stream:
            try:
                metadata = yaml.load(stream)

            except yaml.YAMLError as exc:
                print(exc)

    return BenchmarkRunResource(path, tasks, metadata)


def _benchmark_runs_from_dir(benchmark_dir):
    run_paths = [os.path.join(benchmark_dir, path) for path in os.listdir(benchmark_dir)]
    run_paths = [path for path in run_paths if os.path.isdir(path)]
    return [load_bmrun_from_path(path) for path in run_paths]


def populate_benchmark_cache():
    benchmark_dir = BENCHMARK_VIEWER_DATA_PATH
    logger.info("Loading in all benchmark_runs from %s..." % benchmark_dir)
    bmruns = _benchmark_runs_from_dir(benchmark_dir)

    logger.info("Found %s benchmark_runs in %s. Computing scores for each task..." % (
        len(bmruns), BENCHMARK_VIEWER_DATA_PATH))
    for run in bmruns:
        for task in run.tasks:
            for evaluation in task.evaluations:
                score_cache.cache_evaluation_score(run, task, evaluation.score)

        logger.info("Computed scores for %s" % run.name)


if __name__ == '__main__':
    logger.setLevel(logging.INFO)

    populate_benchmark_cache()

    if ARGS.open:
        subprocess.check_call('open http://localhost:5000', shell=True)

    register_template_helpers(app)
    app.run(debug=ARGS.debug, port=5000)
