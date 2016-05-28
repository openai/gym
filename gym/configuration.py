import logging
import sys

import gym

logger = logging.getLogger(__name__)

root_logger = logging.getLogger()
requests_logger = logging.getLogger('requests')

# Set up the default handler
formatter = logging.Formatter('[%(asctime)s] %(message)s')
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)

# We need to take in the gym logger explicitly since this is called
# at initialization time.
def logger_setup(gym_logger):
    root_logger.addHandler(handler)
    gym_logger.setLevel(logging.INFO)
    # When set to INFO, this will print out the hostname of every
    # connection it makes.
    # requests_logger.setLevel(logging.WARN)

def undo_logger_setup():
    """Undoes the automatic logging setup done by OpenAI Gym. You should call
    this function if you want to manually configure logging
    yourself. Typical usage would involve putting something like the
    following at the top of your script:

    gym.undo_logger_setup()
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stderr))
    """
    root_logger.removeHandler(handler)
    gym.logger.setLevel(logging.NOTSET)
    requests_logger.setLevel(logging.NOTSET)
