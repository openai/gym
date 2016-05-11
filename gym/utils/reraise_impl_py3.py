# http://stackoverflow.com/a/33822606 -- `from None` disables Python 3'
# semi-smart exception chaining, which we don't want in this case.
def reraise_impl(e, traceback):
    raise e.with_traceback(traceback) from None
