import re
import copy
import importlib

from gym import error, logger

# This format is true today, but it's *not* an official spec.
# [username/](env-name)-v(version)    env-name is group 1, version is group 2
#
# 2016-10-31: We're experimentally expanding the environment ID format
# to include an optional username.
env_id_re = re.compile(r"^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$")


def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class EnvSpec(object):
    """A specification for a particular instance of the environment. Used
    to register the parameters for official evaluations.

    Args:
        id (str): The official environment ID
        entry_point (Optional[str]): The Python entrypoint of the environment class (e.g. module.name:Class)
        reward_threshold (Optional[int]): The reward threshold before the task is considered solved
        nondeterministic (bool): Whether this environment is non-deterministic even after seeding
        max_episode_steps (Optional[int]): The maximum number of steps that an episode can consist of
        kwargs (dict): The kwargs to pass to the environment class

    """

    def __init__(
        self,
        id,
        entry_point=None,
        reward_threshold=None,
        nondeterministic=False,
        max_episode_steps=None,
        kwargs=None,
    ):
        self.id = id
        self.entry_point = entry_point
        self.reward_threshold = reward_threshold
        self.nondeterministic = nondeterministic
        self.max_episode_steps = max_episode_steps
        self._kwargs = {} if kwargs is None else kwargs

        match = env_id_re.search(id)
        if not match:
            raise error.Error(
                "Attempted to register malformed environment ID: {}. (Currently all IDs must be of the form {}.)".format(
                    id, env_id_re.pattern
                )
            )
        self._env_name = match.group(1)

    def make(self, **kwargs):
        """Instantiates an instance of the environment with appropriate kwargs"""
        if self.entry_point is None:
            raise error.Error(
                "Attempting to make deprecated env {}. (HINT: is there a newer registered version of this env?)".format(
                    self.id
                )
            )
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            env = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            env = cls(**_kwargs)

        # Make the environment aware of which spec it came from.
        spec = copy.deepcopy(self)
        spec._kwargs = _kwargs
        env.unwrapped.spec = spec

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

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            logger.info("Making new env: %s (%s)", path, kwargs)
        else:
            logger.info("Making new env: %s", path)
        spec = self.spec(path)
        env = spec.make(**kwargs)
        if env.spec.max_episode_steps is not None:
            from gym.wrappers.time_limit import TimeLimit

            env = TimeLimit(env, max_episode_steps=env.spec.max_episode_steps)
        return env

    def all(self):
        return self.env_specs.values()

    def spec(self, path):
        if ":" in path:
            mod_name, _, id = path.partition(":")
            try:
                importlib.import_module(mod_name)
            except ModuleNotFoundError:
                raise error.Error(
                    "A module ({}) was specified for the environment but was not found, make sure the package is installed with `pip install` before calling `gym.make()`".format(
                        mod_name
                    )
                )
        else:
            id = path

        match = env_id_re.search(id)
        if not match:
            raise error.Error(
                "Attempted to look up malformed environment ID: {}. (Currently all IDs must be of the form {}.)".format(
                    id.encode("utf-8"), env_id_re.pattern
                )
            )

        try:
            return self.env_specs[id]
        except KeyError:
            # Parse the env name and check to see if it matches the non-version
            # part of a valid env (could also check the exact number here)
            env_name = match.group(1)
            matching_envs = [
                valid_env_name
                for valid_env_name, valid_env_spec in self.env_specs.items()
                if env_name == valid_env_spec._env_name
            ]
            algorithmic_envs = [
                "Copy",
                "RepeatCopy",
                "DuplicatedInput",
                "Reverse",
                "ReversedAdiiton",
                "ReversedAddition3",
            ]
            toytext_envs = [
                "KellyCoinflip",
                "KellyCoinflipGeneralized",
                "NChain",
                "Roulette",
                "GuessingGame",
                "HotterColder",
            ]
            if matching_envs:
                raise error.DeprecatedEnv(
                    "Env {} not found (valid versions include {})".format(
                        id, matching_envs
                    )
                )
            elif env_name in algorithmic_envs:
                raise error.UnregisteredEnv(
                    "Algorithmic environment {} has been moved out of Gym. Install it via `pip install gym-algorithmic` and add `import gym_algorithmic` before using it.".format(
                        id
                    )
                )
            elif env_name in toytext_envs:
                raise error.UnregisteredEnv(
                    "Toytext environment {} has been moved out of Gym. Install it via `pip install gym-legacy-toytext` and add `import gym_toytext` before using it.".format(
                        id
                    )
                )
            else:
                raise error.UnregisteredEnv("No registered env with id: {}".format(id))

    def register(self, id, **kwargs):
        if id in self.env_specs:
            logger.warn("Overriding environment {}".format(id))
        self.env_specs[id] = EnvSpec(id, **kwargs)


# Have a global registry
registry = EnvRegistry()


def register(id, **kwargs):
    return registry.register(id, **kwargs)


def make(id, **kwargs):
    return registry.make(id, **kwargs)


def spec(id):
    return registry.spec(id)
