import functools
import logging
import os
import random
import sys
import time

from gym import error

logger = logging.getLogger(__name__)

def utf8(value):
    if isinstance(value, unicode) and sys.version_info < (3, 0):
        return value.encode('utf-8')
    else:
        return value

def file_size(f):
    return os.fstat(f.fileno()).st_size

def retry_exponential_backoff(f, errors, max_retries=5, interval=1):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        num_retries = 0
        caught_errors = []
        while True:
            try:
                result = f(*args, **kwargs)
            except errors as e:
                logger.error("Caught error in %s: %s" % (f.__name__, e))
                caught_errors.append(e)

                if num_retries < max_retries:
                    backoff = random.randint(1, 2 ** num_retries) * interval
                    logger.error("Retrying in %.1fs..." % backoff)
                    time.sleep(backoff)
                    num_retries += 1
                else:
                    msg = "Exceeded allowed retries. Here are the individual error messages:\n\n"
                    msg += "\n\n".join("%s: %s" % (type(e).__name__, str(e)) for e in caught_errors)
                    raise error.RetriesExceededError(msg)
            else:
                break
        return result
    return wrapped
