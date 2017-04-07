from __future__ import division

import logging

logger = logging.getLogger(__name__)

import numpy as np


def _rescale_value(value, min, max):
    denom = max - min
    return ((value - min) / denom) if denom != 0.0 else 1.0


def compute_task_score(task_spec, score_cache, evaluations):
    """
    Return the task's score as a mean of all the evaluation scores.
    Requires the score cache because we fill in missing evaluation_scores with the worst scores that
    we have seen.
    """
    env_id = task_spec.env_id
    eval_scores = [evaluation.score
        for evaluation in evaluations
        if evaluation.env_id == env_id
    ]

    # If the benchmark run did not give us sufficient trials, fill it out with the worst ones
    # we've seen so far
    num_required_trials = task_spec.trials
    for missing_trial in range(len(eval_scores), num_required_trials):
        eval_scores.append(score_cache.min_score(env_id))

    return np.mean(eval_scores)


def compute_task_rank(task_spec, score_cache, evaluations):
    """
    Return the task rank as a fraction of the best one we've seen, with 1.0 being
    the best, and 0.0 being the worst
    """
    task_score = compute_task_score(task_spec, score_cache, evaluations)
    best_score_seen = score_cache.max_score(task_spec.env_id)
    worst_score_seen = score_cache.min_score(task_spec.env_id)
    return _rescale_value(task_score, worst_score_seen, best_score_seen)


def compute_benchmark_run_rank(benchmark, score_cache, evaluations):
    """The mean task_rank across all tasks in the benchmark"""
    task_ranks = [compute_task_rank(task_spec, score_cache, evaluations)
        for task_spec in benchmark.tasks]

    return np.mean(task_ranks)
