from gym.benchmarks import ranking


def display_task_score(bmrun, task_spec, score_cache):
    score = score_cache.get_task_score(bmrun, task_spec)
    if score is not None:
        return "%.2f" % score
    else:
        return "N/A"


def display_task_rank(bmrun, task_spec, score_cache):
    evaluations = bmrun.task_by_env_id(task_spec.env_id).evaluations
    rank = ranking.compute_task_rank(task_spec, score_cache, evaluations)
    return "%.2f" % rank


def register_template_helpers(app):
    app.jinja_env.globals.update(
        display_task_score=display_task_score,
        display_task_rank=display_task_rank,
    )
