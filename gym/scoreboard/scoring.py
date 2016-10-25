"""This is the actual code we use to score people's solutions
server-side. The interfaces here are not yet stable, but we include
them so that people can reproduce our scoring calculations
independently.

We correspondly do not currently import this module.
"""

import numpy as np
import requests

import gym

def score_from_remote(url):
    result = requests.get(url)
    parsed = result.json()
    episode_lengths = parsed['episode_lengths']
    episode_rewards = parsed['episode_rewards']
    episode_types = parsed.get('episode_types')
    timestamps = parsed['timestamps']
    # Handle legacy entries where initial_reset_timestamp wasn't set
    initial_reset_timestamp = parsed.get('initial_reset_timestamp', timestamps[0])
    env_id = parsed['env_id']

    spec = gym.spec(env_id)
    return score_from_merged(episode_lengths, episode_rewards, episode_types, timestamps, initial_reset_timestamp, spec.trials, spec.reward_threshold)

def score_from_local(directory):
    """Calculate score from a local results directory"""
    results = gym.monitoring.monitor.load_results(directory)
    # No scores yet saved
    if results is None:
        return None

    episode_lengths = results['episode_lengths']
    episode_rewards = results['episode_rewards']
    episode_types = results['episode_types']
    timestamps = results['timestamps']
    initial_reset_timestamp = results['initial_reset_timestamp']
    spec = gym.spec(results['env_info']['env_id'])

    return score_from_merged(episode_lengths, episode_rewards, episode_types, timestamps, initial_reset_timestamp, spec.trials, spec.reward_threshold)

def score_from_merged(episode_lengths, episode_rewards, episode_types, timestamps, initial_reset_timestamp, trials, reward_threshold):
    """Method to calculate the score from merged monitor files. Scores
    only a single environment; mostly legacy.
    """
    if episode_types is not None:
        # Select only the training episodes
        episode_types = np.array(episode_types)
        t_idx = np.where(episode_types == 't')
        episode_lengths = np.array(episode_lengths)[t_idx]
        episode_rewards = np.array(episode_rewards)[t_idx]
        timestamps = np.array(timestamps)[t_idx]

    # Make sure everything is a float -- no pesky ints.
    episode_rewards = np.array(episode_rewards, dtype='float64')

    episode_t_value = timestep_t_value = mean = error = None
    seconds_to_solve = seconds_in_total = None

    if len(timestamps) > 0:
        # This is: time from the first reset to the end of the last episode
        seconds_in_total = timestamps[-1] - initial_reset_timestamp
    if len(episode_rewards) >= trials:
        means = running_mean(episode_rewards, trials)
        if reward_threshold is not None:
            # Compute t-value by finding the first index at or above
            # the threshold. It comes out as a singleton tuple.
            (indexes_above_threshold, ) = np.where(means >= reward_threshold)
            if len(indexes_above_threshold) > 0:
                # Grab the first episode index that is above the threshold value
                episode_t_value = indexes_above_threshold[0]

                # Find timestep corresponding to this episode
                cumulative_timesteps = np.cumsum(np.insert(episode_lengths, 0, 0))
                # Convert that into timesteps
                timestep_t_value = cumulative_timesteps[episode_t_value]
                # This is: time from the first reset to the end of the first solving episode
                seconds_to_solve = timestamps[episode_t_value] - initial_reset_timestamp

        # Find the window with the best mean
        best_idx = np.argmax(means)
        best_rewards = episode_rewards[best_idx:best_idx+trials]
        mean = np.mean(best_rewards)
        if trials == 1: # avoid NaN
            error = 0.
        else:
            error = np.std(best_rewards) / (np.sqrt(trials) - 1)
    return {
        'episode_t_value': episode_t_value,
        'timestep_t_value': timestep_t_value,
        'mean': mean,
        'error': error,
        'number_episodes': len(episode_rewards),
        'number_timesteps': sum(episode_lengths),
        'seconds_to_solve': seconds_to_solve,
        'seconds_in_total': seconds_in_total,
    }

def benchmark_score_from_merged(benchmark, env_id, episode_lengths, episode_rewards, episode_types):
    """Method to calculate an environment's benchmark score from merged
    monitor files.
    """
    return benchmark.score(benchmark, env_id, episode_lengths, episode_rewards, episode_types)

def benchmark_aggregate_results(benchmark, env_id_to_benchmark_results):
    scores = {}
    solves = {}
    start_times = []
    end_times = []

    scorer = benchmark.scorer

    # N.B. for each env_id, our benchmark_results will have a list of scores,
    # solves, and times corresponding to the different tasks for that env_id. If
    # we don't have enough trials, we zero out the score.
    # TODO could do smarter matching of results to trials if we have extras
    for env_id in benchmark.env_ids:
        # TODO for now, baked in assumption that the number of trials is the
        # same for all tasks involving a particular env.
        task_list = benchmark.task_specs(env_id)
        num_trials = task_list[0].trials
        benchmark_results = env_id_to_benchmark_results[env_id]
        for trial in range(num_trials):
            if trial < len(benchmark_results):
                # okay process this benchmark result against this trial
                benchmark_result = benchmark_results[trial]

                env_scores = scores.setdefault(env_id, [])
                env_scores.append(benchmark_result['scores'])

                # note: solves is a list of lists - for each task for this env,
                # does each episode solve that task. We consider the env solved
                # if every episode for every task is individually solved.
                solved = solves.setdefault(env_id, True)
                solves[env_id] = solved and np.all(benchmark_result['solves'])

                # these timestamps are a list of the first / last valid timestamp
                # for each task involving this env.
                start_times.append(benchmark_result['initial_reset_timestamp'])
                end_times.append(max(benchmark_result['timestamps']))
            else:
                # no matching benchmark result for this trial
                env_scores = scores.setdefault(env_id, [])
                env_scores.append([scorer.null_score for _ in task_list])
                solves[env_id] = False


    score = benchmark.score_benchmark(scores)
    num_envs_solved = len([s for s in solves.values() if s])
    start_to_finish_seconds = max(end_times) - min(start_times)
    summed_training_seconds = np.sum([end - start for end, start in zip(end_times, start_times)])

    return dict(
        score=score,
        num_envs_solved=num_envs_solved,
        start_to_finish_seconds=start_to_finish_seconds,
        summed_training_seconds=summed_training_seconds,
    )

def running_mean(x, N):
    x = np.array(x, dtype='float64')
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

def compute_graph_stats(episode_lengths, episode_rewards, timestamps, initial_reset_timestamp, buckets):
    """Method to compute the aggregates for the graphs."""
    # Not a dependency of OpenAI Gym generally.
    import scipy.stats

    num_episodes = len(episode_lengths)

    # Catch for if no files written which causes error with scipy.stats.binned_statistic
    if num_episodes == 0:
        return None

    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)

    # The index of the start of each episode
    x_timestep = np.cumsum(np.insert(episode_lengths, 0, 0))[:-1]
    assert len(x_timestep) == num_episodes

    # Delta since the beginning of time
    x_seconds = [timestamp - initial_reset_timestamp for timestamp in timestamps]

    # The index of each episode
    x_episode = range(num_episodes)

    # Calculate the appropriate x/y statistics
    x_timestep_y_reward = scipy.stats.binned_statistic(x_timestep, episode_rewards, 'median', buckets)
    x_timestep_y_length = scipy.stats.binned_statistic(x_timestep, episode_lengths, 'median', buckets)

    x_episode_y_reward = scipy.stats.binned_statistic(x_episode, episode_rewards, 'median', buckets)
    x_episode_y_length = scipy.stats.binned_statistic(x_episode, episode_lengths, 'median', buckets)

    x_seconds_y_reward = scipy.stats.binned_statistic(x_seconds, episode_rewards, 'median', buckets)
    x_seconds_y_length = scipy.stats.binned_statistic(x_seconds, episode_lengths, 'median', buckets)

    return {
        'initial_reset_timestamp': initial_reset_timestamp,
        'x_timestep_y_reward': graphable_binned_statistic(x_timestep_y_reward),
        'x_timestep_y_length': graphable_binned_statistic(x_timestep_y_length),
        'x_episode_y_reward': graphable_binned_statistic(x_episode_y_reward),
        'x_episode_y_length': graphable_binned_statistic(x_episode_y_length),
        'x_seconds_y_length': graphable_binned_statistic(x_seconds_y_length),
        'x_seconds_y_reward': graphable_binned_statistic(x_seconds_y_reward),
    }

def graphable_binned_statistic(binned):
    x = running_mean(binned.bin_edges, 2)
    y = binned.statistic
    assert len(x) == len(y)

    # Get rid of nasty NaNs
    valid = np.logical_not(np.isnan(x)) & np.logical_not(np.isnan(y))
    x = x[valid]
    y = y[valid]

    return {
        'x': x,
        'y': y,
    }
