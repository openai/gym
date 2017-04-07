from __future__ import division

import logging

logger = logging.getLogger(__name__)

import numpy as np


def _rescale_value(value, min, max):
    denom = max - min
    return ((value - min) / denom) if denom != 0.0 else 1.0


def compute_task_rank(task_spec, score_cache, evaluations):
    """
    Return the task rank as a fraction of the best one we've seen, with 1.0 being
    the best, and 0.0 being the worst
    """
    env_id = task_spec.env_id
    best_score_seen = score_cache.max_score(task_spec)
    worst_score_seen = score_cache.min_score(task_spec)

    eval_mean_aucs = [evaluation.mean_auc
        for evaluation in evaluations
        if evaluation.env_id == env_id
    ]

    # If the benchmark run did not give us sufficient trials, fill it out with the worst ones
    # we've seen so far
    num_required_trials = task_spec.trials
    for missing_trial in range(len(eval_mean_aucs), num_required_trials):
        eval_mean_aucs.append(worst_score_seen)

    task_score = np.mean(eval_mean_aucs)
    return _rescale_value(task_score, worst_score_seen, best_score_seen)


def compute_benchmark_run_rank(benchmark, score_cache, evaluations):
    """The mean task_rank across all tasks in the benchmark"""
    task_ranks = [compute_task_rank(task_spec, score_cache, evaluations)
        for task_spec in benchmark.tasks]

    return np.mean(task_ranks)
