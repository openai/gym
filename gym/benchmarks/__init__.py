# EXPERIMENTAL: all may be removed soon

import numpy as np

from gym.benchmarks import scoring
from gym.benchmarks.registration import register_benchmark, benchmark_spec, registry

register_benchmark(
    id='Atari7Pixel-v0',
    scorer=scoring.ClipTo01ThenAverage(),
    description='7 Atari games, with pixel observations',
    task_groups={
        'BeamRider-v0': [{
            'seeds': 1,
            'timesteps': 10000000
        }],
        'Breakout-v0': [{
            'seeds': 1,
            'timesteps': 10000000
        }],
        'Enduro-v0': [{
            'seeds': 1,
            'timesteps': 10000000
        }],
        'Pong-v0': [{
            'seeds': 1,
            'timesteps': 10000000
        }],
        'Qbert-v0': [{
            'seeds': 1,
            'timesteps': 10000000
        }],
        'Seaquest-v0': [{
            'seeds': 1,
            'timesteps': 10000000
        }],
        'SpaceInvaders-v0': [{
            'seeds': 1,
            'timesteps': 10000000
        }],
    })

register_benchmark(
    id='Atari7Ram-v0',
    description='7 Atari games, with RAM observations',
    scorer=scoring.ClipTo01ThenAverage(),
    task_groups={
        'BeamRider-ram-v0': [{
            'seeds': 1,
            'timesteps': 10000000
        }],
        'Breakout-ram-v0': [{
            'seeds': 1,
            'timesteps': 10000000
        }],
        'Enduro-ram-v0': [{
            'seeds': 1,
            'timesteps': 10000000
        }],
        'Pong-ram-v0': [{
            'seeds': 1,
            'timesteps': 10000000
        }],
        'Qbert-ram-v0': [{
            'seeds': 1,
            'timesteps': 10000000
        }],
        'Seaquest-ram-v0': [{
            'seeds': 1,
            'timesteps': 10000000
        }],
        'SpaceInvaders-ram-v0': [{
            'seeds': 1,
            'timesteps': 10000000
        }],
    })

register_benchmark(
    id='ClassicControl2-v0',
    description='Simple classic control benchmark',
    scorer=scoring.ClipTo01ThenAverage(),
    task_groups={
        'CartPole-v0': [{
            'seeds': 1,
            'timesteps': 2000,
        }],
        'Pendulum-v0': [{
            'seeds': 1,
            'timesteps': 1000,
        }],
    })
