from gym.benchmarks import ranking


def display_task_score(task_run, score_cache):
    rank = ranking.compute_task_score(task_run.spec, score_cache, task_run.evaluations)
    return "%.2f" % rank


def display_task_rank(task_run, score_cache):
    rank = ranking.compute_task_rank(task_run.spec, score_cache, task_run.evaluations)
    return "%.3f" % rank


def display_bmrun_rank(bmrun, benchmark_spec, score_cache):
    rank = ranking.compute_benchmark_run_rank(benchmark_spec, score_cache, bmrun.evaluations)
    return "%.3f" % rank


def register_template_helpers(app):
    app.jinja_env.globals.update(
        display_task_score=display_task_score,
        display_task_rank=display_task_rank,
        display_bmrun_rank=display_bmrun_rank,
    )
