import cStringIO
import json
import os

from flask import Flask
from gym import monitoring
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


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
        img_bytes = cStringIO.StringIO()
        plt.savefig(img_bytes, format='svg')
        return img_bytes.getvalue()


class Trial(object):

    def __init__(self, episode_rewards, episode_lengths):
        self.episode_rewards = episode_rewards
        self.episode_lengths = episode_lengths

    @classmethod
    def from_stats_file(cls, stats_file):
        with open(stats_file) as f:
            stats = json.load(f)
            return Trial(stats['episode_rewards'], stats['episode_lengths'])


app = Flask(__name__)


@app.route('/')
def view_tasks():
    tasks = {}
    for root, _, fnames in os.walk('/tmp'):
        for fname in fnames:
            if not fname.endswith('manifest.json'):
                continue
            with open(os.path.join(root, fname)) as f:
                manifest = json.load(f)
                env_id = manifest['env_info']['env_id']
                if env_id not in tasks:
                    tasks[env_id] = Task(env_id, [])
                task = tasks[env_id]
                stats_file = os.path.join(root, manifest['stats'])
                task.trials.append(Trial.from_stats_file(stats_file))

    svgs = [task.to_svg() for task in tasks.values()]
    rows = ''.join(
        '<tr><td>{}</td><td>{}</td></tr>'.format(env_id, task.to_svg())
        for env_id, task in sorted(tasks.items())
    )
    return '<table>{}</tbody>'.format(rows)


if __name__ == '__main__':
    app.run(debug=True, port=5030)
