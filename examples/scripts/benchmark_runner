#!/usr/bin/env python
#
# Run all the tasks on a benchmark using a random agent.
#
# This script assumes you have set an OPENAI_GYM_API_KEY environment
# variable. You can find your API key in the web interface:
# https://gym.openai.com/settings/profile.
#
import argparse
import logging
import os
import sys

import gym
# In modules, use `logger = logging.getLogger(__name__)`
from gym import wrappers
from gym.scoreboard.scoring import benchmark_score_from_local

import openai_benchmark

logger = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-b', '--benchmark-id', help='id of benchmark to run e.g. Atari7Ram-v0')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-f', '--force', action='store_true', dest='force', default=False)
    parser.add_argument('-t', '--training-dir', default="/tmp/gym-results", help='What directory to upload.')
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    benchmark_id = args.benchmark_id
    if benchmark_id is None:
        logger.info("Must supply a valid benchmark")
        return 1

    try:
        benchmark = gym.benchmark_spec(benchmark_id)
    except Exception:
        logger.info("Invalid benchmark")
        return 1

    # run benchmark tasks
    for task in benchmark.tasks:
        logger.info("Running on env: {}".format(task.env_id))
        for trial in range(task.trials):
            env = gym.make(task.env_id)
            training_dir_name = "{}/{}-{}".format(args.training_dir, task.env_id, trial)
            env = wrappers.Monitor(env, training_dir_name, video_callable=False, force=args.force)
            env.reset()
            for _ in range(task.max_timesteps):
                o, r, done, _ = env.step(env.action_space.sample())
                if done:
                    env.reset()
            env.close()

    logger.info("""Computing statistics for this benchmark run...
{{
    score: {score},
    num_envs_solved: {num_envs_solved},
    summed_training_seconds: {summed_training_seconds},
    start_to_finish_seconds: {start_to_finish_seconds},
}}

    """.rstrip().format(**benchmark_score_from_local(benchmark_id, args.training_dir)))

    logger.info("""Done running, upload results using the following command:

python -c "import gym; gym.upload('{}', benchmark_id='{}', algorithm_id='(unknown)')"

    """.rstrip().format(args.training_dir, benchmark_id))

    return 0

if __name__ == '__main__':
    sys.exit(main())
