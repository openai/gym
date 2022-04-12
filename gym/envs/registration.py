import contextlib
import importlib
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Type, Union

if sys.version_info < (3, 10):
    import importlib_metadata as metadata  # type: ignore
else:
    import importlib.metadata as metadata

from gym import error, logger

ENV_ID_RE: re.Pattern = re.compile(
    r"^(?:(?P<namespace>[\w:-]+)\/)?(?:(?P<name>[\w:.-]+?))(?:-v(?P<version>\d+))?$"
)


def load(name: str) -> Type:
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def parse_env_id(id: str) -> Tuple[Optional[str], str, Optional[int]]:
    """Parse environment ID string format.

    This format is true today, but it's *not* an official spec.
    [username/](env-name)-v(version)    env-name is group 1, version is group 2

    2016-10-31: We're experimentally expanding the environment ID format
    to include an optional username.
    """
    match = ENV_ID_RE.fullmatch(id)
    if not match:
        raise error.Error(
            f"Malformed environment ID: {id}."
            f"(Currently all IDs must be of the form {ENV_ID_RE}.)"
        )
    namespace, name, version = match.group("namespace", "name", "version")
    if version is not None:
        version = int(version)

    return namespace, name, version


@dataclass
class EnvSpec:
    id: str
    entry_point: Optional[Union[Callable, str]] = field(default=None)
    reward_threshold: Optional[float] = field(default=None)
    nondeterministic: bool = field(default=False)
    max_episode_steps: Optional[int] = field(default=None)
    order_enforce: bool = field(default=True)
    kwargs: dict = field(default_factory=dict)

    namespace: Optional[str] = field(init=False)
    name: str = field(init=False)
    version: Optional[int] = field(init=False)

    def __post_init__(self):
        # Initialize namespace, name, version
        self.namespace, self.name, self.version = parse_env_id(self.id)


# Global registry of environments. Meant to be accessed through `register` and `make`
registry = dict()


def register(id: str, **kwargs):
    spec = EnvSpec(id=id, **kwargs)
    if spec.id in registry:
        raise error.Error(
            f"Attempted to register {spec.id} but it was already registered"
        )
    registry[spec.id] = spec


def make(env_id: str, **kwargs):
    spec = registry.get(env_id)
    if spec is None:
        # TODO: make this a bit more specific
        raise error.Error(f"No registered env with id: {env_id}")

    # TODO: add a minimal env checker on initialization
    if spec.entry_point is None:
        raise error.Error(f"{spec.id} registered but entry_point is not specified")
    elif callable(spec.entry_point):
        return spec.entry_point(**kwargs)
    else:
        # Assume it's a string
        cls = load(spec.entry_point)
        return cls(**kwargs)


def _find_newest_version(env_id: str) -> Optional[str]:
    env_ids = [env_id_ for env_id_ in registry.keys() if env_id_.startswith(env_id)]
    if not env_ids:
        return None
    elif len(env_ids) == 1:
        return env_ids[0]
    else:
        return max(
            env_ids,
            key=lambda env_id: parse_env_id(env_id)[2],
        )


def load_env_plugins(entry_point: str = "gym.envs") -> None:
    # Load third-party environments
    for plugin in metadata.entry_points(group=entry_point):
        # Python 3.8 doesn't support plugin.module, plugin.attr
        # So we'll have to try and parse this ourselves
        try:
            module, attr = plugin.module, plugin.attr  # type: ignore  ## error: Cannot access member "attr" for type "EntryPoint"
        except AttributeError:
            if ":" in plugin.value:
                module, attr = plugin.value.split(":", maxsplit=1)
            else:
                module, attr = plugin.value, None
        except:
            module, attr = None, None
        finally:
            if attr is None:
                raise error.Error(
                    f"Gym environment plugin `{module}` must specify a function to execute, not a root module"
                )

        fn = plugin.load()
        try:
            fn()
        except Exception as e:
            logger.warn(str(e))
