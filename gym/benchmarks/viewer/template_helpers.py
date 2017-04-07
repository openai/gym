def display_task_score(bmrun, task_spec, score_cache):
    score = score_cache.get_task_score(bmrun, task_spec)
    if score is not None:
        return "%.2f" % score
    else:
        return "N/A"


def register_template_helpers(app):
    app.jinja_env.globals.update(
        display_task_score=display_task_score
    )
