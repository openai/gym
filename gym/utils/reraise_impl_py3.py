def reraise_impl(e, traceback):
    raise e.with_traceback(traceback)
