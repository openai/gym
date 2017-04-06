import json
import os

import io
from flask import Flask
from gym import monitoring
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from gym.benchmarks import registry

BENCHMARK_ID = 'AtariExploration40M'

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

    def __init__(self, env_id, trials):
        self.env_id = env_id
        self.trials = trials

    def to_svg(self):
        plt.figure()
        plt.rcParams['figure.figsize'] = (15, 2)
        for trial in self.trials:
            xs, ys = smooth_reward_curve(
                trial.episode_rewards, trial.episode_lengths, 1e6)
            plt.plot(xs, ys)

        plt.xlabel('Time')
        plt.ylabel('Rewards')
        plt.tight_layout()
        img_bytes = io.StringIO()
        plt.savefig(img_bytes, format='svg')
        return img_bytes.getvalue()


class Trial(object):
    def __init__(self, results):
        self.env_id = results['env_info']['env_id']

        self.episode_rewards = results['episode_rewards']
        self.episode_lengths = results['episode_lengths']
        self.episode_types = results['episode_types']
        self.timestamps = results['timestamps']
        self.initial_reset_timestamps = results['initial_reset_timestamps']

    @classmethod
    def from_training_dir(cls, training_dir):
        results = monitoring.load_results(training_dir)
        return Trial(results)

    def score(self):
        benchmark = registry.benchmark_spec(BENCHMARK_ID)

        fake_data_sources = [0] * len(self.episode_lengths)

        return benchmark.score_evaluation(
            self.env_id,
            data_sources=fake_data_sources,
            initial_reset_timestamps=self.initial_reset_timestamps,
            episode_lengths= self.episode_lengths,
            episode_rewards=self.episode_rewards,
            episode_types=self.episode_types,
            timestamps=self.timestamps)

class Run(object):
    def __init__(self, trials):
        self.trials = trials

app = Flask(__name__)





@app.route('/')
def index():
    run_paths = os.listdir('/tmp/{}'.format(BENCHMARK_ID))

    for run_path in run_paths:
        tasks_from_run_path(run_path)
    # Compute best and worst performance on each task

    # Compute rank for each of them

    # Show them in a list

    pass

@app.route('/compare/<run_name>/<other_run_name>/')
def compare(run_name, other_run_name):
    pass

@app.route('/run/<run_name>')
def view_tasks(run_name):
    tasks = tasks_from_run_path('/tmp/{}/{}'.format(BENCHMARK_ID, run_name))


    rows = ''.join(
        '<tr><td>{}</td><td>{}</td></tr>'.format(env_id, task.to_svg())
        for env_id, task in sorted(tasks.items())
    )
    return '<table>{}</tbody>'.format(rows)


def tasks_from_run_path(path):
    """
    Returns a map of env_ids to tasks included in the run at the path
    """
    env_id_to_task = {}
    for root, _, fnames in os.walk(path):
        for fname in fnames:
            if not fname.endswith('manifest.json'):
                continue

            training_dir = os.path.dirname(fname)
            trial = Trial.from_training_dir(training_dir)

            env_id = trial.env_id

            if env_id not in env_id_to_task:
                env_id_to_task[env_id] = Task(env_id, [])
            task = env_id_to_task[env_id]

            print(trial.score())
            task.trials.append(trial)

    return env_id_to_task


if __name__ == '__main__':
    app.run(debug=True, port=5030)
