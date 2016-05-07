import logging
import pkg_resources
import re
import sys
from gym import error

logger = logging.getLogger(__name__)
# This format is true today, but it's *not* an official spec.
env_id_re = re.compile(r'^([\w:-]+)-v(\d+)$')

def load(name):
    entry_point = pkg_resources.EntryPoint.parse('x={}'.format(name))
    result = entry_point.load(False)
    return result

class EnvSpec(object):
    """A specification for a particular instance of the environment. Used
    to register the parameters for official evaluations.

    Args:
        id (str): The official environment ID
        entry_point (Optional[str]): The Python entrypoint of the environment class (e.g. module.name:Class)
        timestep_limit (int): The max number of timesteps per episode during training
        trials (int): The number of trials to average reward over
        reward_threshold (Optional[int]): The reward threshold before the task is considered solved
        kwargs (dict): The kwargs to pass to the environment class

    Attributes:
        id (str): The official environment ID
        timestep_limit (int): The max number of timesteps per episode in official evaluation
        trials (int): The number of trials run in official evaluation
    """

    def __init__(self, id, entry_point=None, timestep_limit=1000, trials=100, reward_threshold=None, kwargs=None):
        self.id = id
        # Evaluation parameters
        self.timestep_limit = timestep_limit
        self.trials = trials
        self.reward_threshold = reward_threshold

        # We may make some of these other parameters public if they're
        # useful.
        match = env_id_re.search(id)
        if not match:
            raise error.Error('Attempted to register malformed environment ID: {}. (Currently all IDs must be of the form {}.)'.format(id, env_id_re.pattern))
        self._entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self):
        """Instantiates an instance of the environment with appropriate kwargs"""
        if self._entry_point is None:
            raise error.Error('Attempting to make deprecated env {}. (HINT: is there a newer registered version of this env?)'.format(self.id))

        cls = load(self._entry_point)
        env = cls(**self._kwargs)

        # Make the enviroment aware of which spec it came from.
        env.spec = self
        return env

    def __repr__(self):
        return "EnvSpec({})".format(self.id)


class EnvRegistry(object):
    """Register an env by ID. IDs remain stable over time and are
    guaranteed to resolve to the same environment dynamics (or be
    desupported). The goal is that results on a particular environment
    should always be comparable, and not depend on the version of the
    code that was running.
    """

    def __init__(self):
        self.env_specs = {}

    def make(self, id):
        logger.info('Making new env: %s', id)
        spec = self.spec(id)
        return spec.make()

    def all(self):
        return self.env_specs.values()

    def spec(self, id):
        match = env_id_re.search(id)
        if not match:
            raise error.Error('Attempted to look up malformed environment ID: {}. (Currently all IDs must be of the form {}.)'.format(id.encode('utf-8'), env_id_re.pattern))

        try:
            return self.env_specs[id]
        except KeyError:
            raise error.UnregisteredEnv('No registered env with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.env_specs:
            raise error.Error('Cannot re-register id: {}'.format(id))
        self.env_specs[id] = EnvSpec(id, **kwargs)

# Have a global registry
registry = EnvRegistry()
register = registry.register
make = registry.make
spec = registry.spec
