from __future__ import division

import logging
import numpy as np
from gym import envs

logger = logging.getLogger(__name__)

def benchmark_aggregate_score(benchmark, env_id_to_benchmark_results):
    scores = {}
    solves = {}
    start_times = []
    end_times = []
    elapsed_times = []

    # N.B. for each env_id, our benchmark_results will have a list of scores,
    # solves, and times corresponding to the different tasks for that env_id. If
    # we don't have enough trials, we zero out the score.
    # TODO could do smarter matching of results to trials if we have extras
    # TODO for now, baked in assumption that the number of trials is the
    # same for all tasks involving a particular env.
    for env_id in benchmark.env_ids:
        task_list = benchmark.task_specs(env_id)
        num_trials = task_list[0].trials
        benchmark_results = env_id_to_benchmark_results.get(env_id, [])
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
                solves[env_id] = solved and np.sum(benchmark_result['solves'])

                # these timestamps are a list of the first / last valid timestamp
                # for each task involving this env.
                start_times.append(benchmark_result['initial_reset_timestamp'])
                end_times.append(max(benchmark_result['timestamps']))
                elapsed_times.extend(benchmark_result['elapsed_times'])
            else:
                # no matching benchmark result for this trial
                # TODOJT bug?
                env_scores = scores.setdefault(env_id, [])
                env_scores.append([benchmark.scorer.null_score for _ in task_list])
                solves[env_id] = False

    score = benchmark.score_benchmark(scores)
    num_envs_solved = len([s for s in solves.values() if s])
    start_to_finish_seconds = max(end_times) - min(start_times) if end_times and start_times else 0.0
    summed_task_wall_time = np.sum([end - start for end, start in zip(end_times, start_times)])
    summed_training_seconds = np.sum(elapsed_times)

    return dict(
        score=score,
        num_envs_solved=num_envs_solved,
        start_to_finish_seconds=start_to_finish_seconds,
        summed_task_wall_time=summed_task_wall_time,
        summed_training_seconds=summed_training_seconds,
    )

class ClipTo01ThenAverage(object):
    """Benchmark scoring rule

    For each task, we take the last num_episodes (default: 100) evaluation
    episodes before either the max_seconds or max_timesteps limit, whichever is
    earlier. If there are not num_episodes evaluations, we fill in the rest with
    scores of reward_floor.

    For each valid evaluation episode, we clip the reward to be between the
    reward_floor and reward_ceiling for that task. The score for the task is the
    average across all episodes.

    The benchmark score is the average of all task scores.

    """
    def __init__(self, num_episodes=100):
        self.num_episodes = num_episodes

    @property
    def null_score(self):
        """
        This is used to compute benchmark scores when we are missing an evaluation
        """
        return 0.0

    def score_evaluation(self, benchmark, env_id, data_sources, initial_reset_timestamps, episode_lengths, episode_rewards, episode_types, timestamps):
        tasks = benchmark.task_specs(env_id)
        spec = envs.spec(env_id)

        #### 0. Compute timing stats

        if len(initial_reset_timestamps) > 0:
            initial_reset_timestamp = min(initial_reset_timestamps)
        else:
            initial_reset_timestamp = 0


        # How long each episode actually took
        # How long each episode actually took
        durations = np.zeros(len(timestamps))

        data_sources = np.array(data_sources)
        timestamps = np.array(timestamps)
        for source, initial_ts in enumerate(initial_reset_timestamps):
            (source_indexes,) = np.where(data_sources == source)

            if len(source_indexes) == 0:
                continue
            # Once we know the indexes corresponding to a particular
            # source (i.e. worker thread), we can just subtract
            # adjoining values
            durations[source_indexes[0]] = timestamps[source_indexes[0]] - initial_ts
            durations[source_indexes[1:]] = timestamps[source_indexes[1:]] - timestamps[source_indexes[:-1]]

        #### 1. Select out which indexes are for evaluation and which are for training

        (t_idx,) = np.where([t == 't' for t in episode_types]) # training episodes
        (e_idx,) = np.where([t == 'e' for t in episode_types]) # evaluation episodes
        if len(e_idx) == 0:
            # If no episodes marked for evaluation, consider
            # everything both a training and evaluation episode.
            (t_idx,) = np.where([True for t in episode_types])
            (e_idx,) = np.where([True for t in episode_types])

        #### 2. Grab the data corresponding to each of evaluation/training

        training_lengths = np.array(episode_lengths)[t_idx]
        training_rewards = np.array(episode_rewards)[t_idx]
        training_durations = np.array(durations)[t_idx]

        evaluation_lengths = np.array(episode_lengths)[e_idx]
        evaluation_rewards = np.array(episode_rewards)[e_idx]
        evaluation_durations = np.array(durations)[e_idx]

        #### 3. Calculate the total elapsed time (in various units)
        #### for each episode

        # How many training timesteps have elapsed by the end of each
        # episode. Not to be confused with Unix timestamps.
        elapsed_timesteps = np.cumsum(training_lengths)
        # Total number of seconds elapsed by the end of each
        # episode. Note that with n parallel workers each running for
        # m seconds, we want to count the total time as n * m.
        elapsed_seconds = np.cumsum(training_durations)

        scores = []
        solves = []
        rewards = []
        lengths = []
        _timestamps = []
        elapsed_times = []
        for task in tasks:
            # Find the first episode where we're over the allotted
            # training timesteps.
            cutoff_idx = np.inf
            if task.max_timesteps:
                # this looks a little funny, but we want the first idx greater
                # than the cutoff
                (timestep_cutoff,) = np.where(elapsed_timesteps > task.max_timesteps)
                if len(timestep_cutoff) > 0:
                    cutoff_idx = min(cutoff_idx, timestep_cutoff[0])
            if task.max_seconds:
                (seconds_cutoff,) = np.where(elapsed_seconds > task.max_seconds)
                if len(seconds_cutoff) > 0:
                    cutoff_idx = min(cutoff_idx, seconds_cutoff[0])
            if np.isfinite(cutoff_idx):
                orig_cutoff_idx = t_idx[cutoff_idx] # cutoff index in the original (i.e. before filtering to training/evaluation)
                (allowed_e_idx,) = np.where(e_idx < orig_cutoff_idx) # restrict to earlier episodes
            else:
                # All episodes are fair game
                allowed_e_idx = e_idx

            # Grab the last num_episodes evaluation episodes from
            # before the cutoff (at which point we've gathered too
            # much experience).
            #
            # This probably won't work long-term but is fine for now.
            allowed_episode_rewards = np.array(episode_rewards)[allowed_e_idx]
            reward = allowed_episode_rewards[-self.num_episodes:]
            allowed_episode_lengths = np.array(episode_lengths)[allowed_e_idx]
            length = allowed_episode_lengths[-self.num_episodes:]

            floor = task.reward_floor
            ceiling = task.reward_ceiling

            if len(reward) < self.num_episodes:
                extra = self.num_episodes-len(reward)
                logger.info('Only %s rewards for %s; adding %s', len(reward), env_id, extra)
                reward = np.concatenate([reward, [floor] * extra])
                length = np.concatenate([length, [0] * extra])

            # Grab the indexes where we reached the ceiling
            solved = reward >= ceiling
            # Linearly rescale rewards to between 0 and 1
            clipped = np.clip((reward - floor) / (ceiling - floor), 0, 1)

            # Take the mean rescaled score
            score = np.mean(clipped)
            scores.append(score)
            # Record the list of solved episodes
            solves.append(solved)
            # Record the list of rewards
            rewards.append(reward)
            # Record the list of lengths
            lengths.append(length)

            if len(allowed_e_idx) > 0:
                if not np.isfinite(cutoff_idx):
                    cutoff_idx = len(elapsed_seconds) - 1
                last_t_idx = t_idx[cutoff_idx]
                # timestamps is full length
                last_timestamp = timestamps[last_t_idx]
                # elapsed seconds contains only training
                elapsed_time = elapsed_seconds[cutoff_idx]
            else:
                # If we don't have any evaluation episodes, then the
                # last valid timestamp is when we started.
                last_timestamp = initial_reset_timestamp
                elapsed_time = 0.0

            # Record the timestamp of the last episode timestamp
            _timestamps.append(last_timestamp)
            elapsed_times.append(elapsed_time)

        return {
            'rewards': rewards,
            'lengths': lengths,
            'scores': scores,
            'solves': solves,
            'timestamps': _timestamps,
            'elapsed_times': elapsed_times,
            'initial_reset_timestamp': initial_reset_timestamp,
        }

    def score_benchmark(self, benchmark, episode_scores):
        all_scores = []
        for env_id, scores in episode_scores.items():
            all_scores += scores

        return np.mean(all_scores)

def _compute_episode_durations(initial_reset_timestamps, data_sources, timestamps):
    # We'd like to compute the actual time taken by each episode.
    # This should be a simple as subtracting adjoining timestamps

    # However all the monitor timestamps are mixed together from multiple
    # sources, so we do some munging to separate out by source the data_source
    # is an array of ints that is the same size as timestamps and maps back to
    # the original source initial_reset_timestamps is an array with the initial
    # timestamp for each source file

    # TODO if we don't merge monitor files together at a higher level this logic
    # can be a lot simpler

    durations = np.zeros(len(timestamps))
    data_sources = np.array(data_sources)
    for source, initial_ts in enumerate(initial_reset_timestamps):
        (source_indexes,) = np.where(data_sources == source)

        if len(source_indexes) == 0:
            continue
        # Once we know the indexes corresponding to a particular
        # source (i.e. worker thread), we can just subtract
        # adjoining values
        durations[source_indexes[0]] = timestamps[source_indexes[0]] - initial_ts
        durations[source_indexes[1:]] = timestamps[source_indexes[1:]] - timestamps[source_indexes[:-1]]
    return durations

def _find_cutoffs_for_task(task, elapsed_timesteps, elapsed_seconds):
    # Apply max_timesteps and max_seconds cutoffs. Return np.inf if no cutoff is necessary
    cutoff_idx = np.inf
    if task.max_timesteps:
        # this looks a little funny, but we want the first idx greater
        # than the cutoff
        (timestep_cutoff,) = np.where(elapsed_timesteps > task.max_timesteps)
        if len(timestep_cutoff) > 0:
            cutoff_idx = min(cutoff_idx, timestep_cutoff[0])
    if task.max_seconds:
        (seconds_cutoff,) = np.where(elapsed_seconds > task.max_seconds)
        if len(seconds_cutoff) > 0:
            cutoff_idx = min(cutoff_idx, seconds_cutoff[0])

    return cutoff_idx

class BenchmarkScoringRule(object):
    """Benchmark scoring rule class

    Takes care of munging the monitor files to identify which episodes for each
    task appear before the max_seconds or max_timesteps limit, whichever is
    earlier.

    It passes the rewards for the episodes to the "score_and_solved_func"
    callback given in __init__

    The benchmark score is the average of all task scores.

    """
    def __init__(self, score_and_solved_func):
        self.score_and_solved_func = score_and_solved_func

    @property
    def null_score(self):
        return 0.0

    def score_evaluation(self, benchmark, env_id, data_sources, initial_reset_timestamps, episode_lengths, episode_rewards, episode_types, timestamps):
        tasks = benchmark.task_specs(env_id)
        spec = envs.spec(env_id)

        #### 0. Compute timing stats

        if len(initial_reset_timestamps) > 0:
            initial_reset_timestamp = min(initial_reset_timestamps)
        else:
            initial_reset_timestamp = 0


        # How long each episode actually took
        timestamps = np.array(timestamps)
        durations = _compute_episode_durations(initial_reset_timestamps, data_sources, timestamps)

        #### Grab the data corresponding to each of evaluation/training
        lengths = np.array(episode_lengths)
        rewards = np.array(episode_rewards)

        #### Calculate the total elapsed time (in various units)
        #### for each episode

        # How many training timesteps have elapsed by the end of each
        # episode. Not to be confused with Unix timestamps.
        elapsed_timesteps = np.cumsum(lengths)
        # Total number of seconds elapsed by the end of each
        # episode. Note that with n parallel workers each running for
        # m seconds, we want to count the total time as n * m.
        elapsed_seconds = np.cumsum(durations)

        # List of score for each task
        scores = []
        # List of lists of solved episodes for each task
        solves = []
        # List of lists of episode rewards for each task
        rewards = []
        # List of lists of relevant episode lengths for each task
        cutoff_lengths = []
        _timestamps = []
        elapsed_times = []
        for task in tasks:
            # Find the first episode where we're over the allotted
            # training timesteps.
            cutoff_idx = _find_cutoffs_for_task(task, elapsed_timesteps, elapsed_seconds)
            if not np.isfinite(cutoff_idx):
                # All episodes are fair game
                cutoff_idx = len(lengths)

            reward = np.array(episode_rewards)[:cutoff_idx]

            score, solved = self.score_and_solved_func(task, reward, elapsed_seconds[:cutoff_idx])

            scores.append(score)
            solves.append(solved)
            rewards.append(reward)
            cutoff_lengths.append(lengths[:cutoff_idx])

            if np.any(timestamps[:cutoff_idx]):
                last_timestamp = timestamps[cutoff_idx - 1]
                elapsed_time = elapsed_seconds[cutoff_idx - 1]
            else:
                # If we don't have any valid episodes, then the
                # last valid timestamp is when we started.
                last_timestamp = initial_reset_timestamp
                elapsed_time = 0.0

            # Record the timestamp of the last episode
            _timestamps.append(last_timestamp)
            elapsed_times.append(elapsed_time)

        return {
            'rewards': rewards,
            'lengths': cutoff_lengths,
            'scores': scores,
            'solves': solves,
            'timestamps': _timestamps,
            'elapsed_times': elapsed_times,
            'initial_reset_timestamp': initial_reset_timestamp,
        }

    def score_benchmark(self, benchmark, episode_scores):
        all_scores = []
        for env_id, scores in episode_scores.items():
            all_scores += scores

        return np.mean(all_scores)


def total_reward_from_episode_rewards(task, reward, elapsed_seconds):
    "TotalReward scoring takes the mean of all rewards earned over the course of the episode and clips it between reward_floor and reward_ceiling"
    # reward is an array containing valid rewards for the episode
    floor = task.reward_floor
    ceiling = task.reward_ceiling

    solved = reward >= ceiling
    # Sum raw rewards, linearly rescale to between 0 and 1
    score = np.clip((np.mean(reward) - floor) / (ceiling - floor), 0, 1)
    return score, solved


class TotalReward(BenchmarkScoringRule):
    def __init__(self):
        super(TotalReward, self).__init__(total_reward_from_episode_rewards)


def reward_per_time_from_episode_rewards(task, reward, elapsed_seconds):
    "RewardPerTime scoring takes the total reward earned over the course of the episode, divides by the elapsed time, and clips it between reward_floor and reward_ceiling"
    floor = task.reward_floor
    ceiling = task.reward_ceiling

    # TODO actually compute solves for this
    solved = np.zeros(len(reward))

    # Sum the rewards for all episodes, divide by total time taken for all episodes
    reward_per_second = np.sum(reward) / elapsed_seconds[-1] if np.any(elapsed_seconds) else 0.0
    score = np.clip((reward_per_second - floor) / (ceiling - floor), 0, 1)
    return score, solved


class RewardPerTime(BenchmarkScoringRule):
    def __init__(self):
        super(RewardPerTime, self).__init__(reward_per_time_from_episode_rewards)
