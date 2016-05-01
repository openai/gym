def reraise_impl(e, traceback):
    raise e.__class__, e, traceback
