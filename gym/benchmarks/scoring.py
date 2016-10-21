import logging
import numpy as np
from gym import envs

logger = logging.getLogger(__name__)

class ClipTo01ThenAverage(object):
    def __init__(self, num_episodes=100):
        self.num_episodes = num_episodes

    def score_evaluation(self, benchmark, env_id, data_sources, initial_reset_timestamps, episode_lengths, episode_rewards, episode_types, timestamps):
        tasks = benchmark.task_specs(env_id)
        spec = envs.spec(env_id)

        #### 0. Compute timing stats

        if len(initial_reset_timestamps) > 0:
            initial_reset_timestamp = min(initial_reset_timestamps)
        else:
            initial_reset_timestamp = 0

        # How long each episode actually took
        durations = np.zeros(len(timestamps))

        # (Details computing duration.)
        data_sources = np.array(data_sources)
        timestamps = np.array(timestamps)
        for source in range(len(initial_reset_timestamps)):
            # Once we know the indexes corresponding to a particular
            # source (i.e. worker thread), we can just subtract
            # adjoining values
            source_indexes = np.where(data_sources == source)

            durations[source_indexes[0]] = timestamps[source_indexes[0]] - initial_reset_timestamps[source]
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
        _timestamps = []
        for task in tasks:
            # Find the first episode where we're over the allotted
            # training timesteps.
            cutoff_idx = np.inf
            if task.max_timesteps:
                (timestep_cutoff,) = np.where(elapsed_timesteps > task.max_timesteps)
                if len(timestep_cutoff) > 0:
                    cutoff_idx = min(cutoff_idx, timestep_cutoff[-1])
            if task.max_seconds:
                (seconds_cutoff,) = np.where(elapsed_seconds > task.max_seconds)
                if len(seconds_cutoff) > 0:
                    cutoff_idx = min(cutoff_idx, seconds_cutoff[-1])
            if np.isfinite(cutoff_idx):
                orig_cutoff_idx = t_idx[cutoff_idx] # cutoff index in the original (i.e. before filtering to training/evaluation)
                (allowed_e_idx,) = np.where(e_idx < orig_cutoff_idx) # restrict to earlier episodes
            else:
                # All episodes are fair game
                allowed_e_idx = e_idx

            if len(allowed_e_idx) > 0:
                last_timestamp = timestamps[allowed_e_idx[-1]]
            else:
                # If we don't have any evaluation episodes, then the
                # last valid timestamp is when we started.
                last_timestamp = initial_reset_timestamp

            # Grab the last num_episodes evaluation episodes from
            # before the cutoff (at which point we've gathered too
            # much experience).
            #
            # This probably won't work long-term but is fine for now.
            allowed_episode_rewards = np.array(episode_rewards)[allowed_e_idx]
            reward = allowed_episode_rewards[-self.num_episodes:]

            floor = task.reward_floor
            ceiling = task.reward_ceiling

            if len(reward) < self.num_episodes:
                extra = self.num_episodes-len(reward)
                logger.info('Only %s rewards for %s; adding %s', len(reward), env_id, extra)
                reward = np.concatenate([reward, [floor] * extra])

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
            # Record the timestamp of the last episode timestamp
            _timestamps.append(last_timestamp)

        return {
            'rewards': rewards,
            'scores': scores,
            'solves': solves,
            'timestamps': _timestamps,
        }

    def score_benchmark(self, benchmark, episode_scores):
        all_scores = []
        for env_id, scores in episode_scores.items():
            all_scores += scores

        return np.mean(all_scores)
