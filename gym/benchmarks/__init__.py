# EXPERIMENTAL: all may be removed soon

from gym.benchmarks.registration import register_benchmark, benchmark_spec

register_benchmark(
    id='Atari7Pixel-v0',
    score_method='average_last_100_episodes',
    tasks={
        "BeamRider": {
            "env_id": "BeamRider-v0",
            "seeds": 1,
            "timesteps": 10000000
        },
        "Breakout": {
            "env_id": "Breakout-v0",
            "seeds": 1,
            "timesteps": 10000000
        },
        "Enduro": {
            "env_id": "Enduro-v0",
            "seeds": 1,
            "timesteps": 10000000
        },
        "Pong": {
            "env_id": "Pong-v0",
            "seeds": 1,
            "timesteps": 10000000
        },
        "Qbert": {
            "env_id": "Qbert-v0",
            "seeds": 1,
            "timesteps": 10000000
        },
        "Seaquest": {
            "env_id": "Seaquest-v0",
            "seeds": 1,
            "timesteps": 10000000
        },
        "SpaceInvaders": {
            "env_id": "SpaceInvaders-v0",
            "seeds": 1,
            "timesteps": 10000000
        }
    })

register_benchmark(
    id='Atari7Ram-v0',
    score_method='average_last_100_episodes',
    tasks={
        "BeamRider": {
            "env_id": "BeamRider-ram-v0",
            "seeds": 1,
            "timesteps": 10000000
        },
        "Breakout": {
            "env_id": "Breakout-ram-v0",
            "seeds": 1,
            "timesteps": 10000000
        },
        "Enduro": {
            "env_id": "Enduro-ram-v0",
            "seeds": 1,
            "timesteps": 10000000
        },
        "Pong": {
            "env_id": "Pong-ram-v0",
            "seeds": 1,
            "timesteps": 10000000
        },
        "Qbert": {
            "env_id": "Qbert-ram-v0",
            "seeds": 1,
            "timesteps": 10000000
        },
        "Seaquest": {
            "env_id": "Seaquest-ram-v0",
            "seeds": 1,
            "timesteps": 10000000
        },
        "SpaceInvaders": {
            "env_id": "SpaceInvaders-ram-v0",
            "seeds": 1,
            "timesteps": 10000000
        }
    })
