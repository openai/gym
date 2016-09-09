import logging
import numpy as np
from gym import envs

logger = logging.getLogger(__name__)

class ClipTo01ThenAverage(object):
    def __init__(self, num_episodes=100):
        self.num_episodes = num_episodes

    def score_evaluation(self, benchmark, env_id, episode_lengths, episode_rewards, episode_types):
        tasks = benchmark.task_groups[env_id]
        spec = envs.spec(env_id)

        (t_idx,) = np.where([t == 't' for t in episode_types]) # training episodes
        (e_idx,) = np.where([t == 'e' for t in episode_types]) # evaluation episodes
        training_lengths = np.array(episode_lengths)[t_idx]
        training_rewards = np.array(episode_rewards)[t_idx]

        evaluation_lengths = np.array(episode_lengths)[e_idx]
        evaluation_rewards = np.array(episode_rewards)[e_idx]

        # How many training timesteps have elapsed by the end of each episode
        elapsed_timesteps = np.cumsum(training_lengths)

        scores = []
        solves = []
        for task in tasks:
            # Find the first episode where we're over the allotted
            # training timesteps.
            (cutoff,) = np.where(elapsed_timesteps > task.timesteps)
            if len(cutoff) > 0:
                cutoff_idx = cutoff[-1]
                orig_cutoff_idx = t_idx[cutoff_idx] # cutoff index in the original
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
            rewards = allowed_episode_rewards[-self.num_episodes:]

            if len(rewards) == 0:
                logger.info('No rewards for %s', env_id)
                scores.append(0)
                return

            floor = task.reward_floor
            ceiling = task.reward_ceiling

            # Grab the indexes where we reached the ceiling
            solved = rewards >= ceiling
            # Linearly rescale rewards to between 0 and 1
            clipped = np.clip((rewards - floor) / (ceiling - floor), 0, 1)

            # Take the mean rescaled score
            score = np.mean(clipped)
            scores.append(score)
            solves.append(solved)

        return scores, solves

    def score_benchmark(self, benchmark, episode_scores):
        all_scores = []
        for env_id, scores in episode_scores.items():
            all_scores += scores

        return np.mean(all_scores)
