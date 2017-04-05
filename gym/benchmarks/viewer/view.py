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



if __name__ == '__main__':
    training_dir = '/tmp/mujoco1m-v0_1489625716_tom_test_kube/tom_test_kube_1489625716_hopper-v1_1/gym'
    results = monitoring.load_results(training_dir)
    print(results.keys())

    rewards = results['episode_rewards']
    lengths = results['episode_lengths']

    xs, ys = smooth_reward_curve(rewards, lengths, 1000000)

    plt.plot(xs, ys)
    plt.ylabel('Rewards over time')
    plt.show()
