import time

_last_time = time.time()


def time_elapsed():
    global _last_time
    now = time.time()
    elapsed = now - _last_time
    _last_time = now

    return "%.3f" % elapsed
