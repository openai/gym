import re
import sys
import copy
import importlib
import contextlib

if sys.version_info < (3, 8):
    import importlib_metadata as metadata
else:
    import importlib.metadata as metadata

from collections import defaultdict
from collections.abc import MutableMapping
from operator import getitem

from typing import Optional, Union, Dict, Set, Tuple, Generator

from gym import error, logger

# This format is true today, but it's *not* an official spec.
# [username/](env-name)-v(version)    env-name is group 1, version is group 2
#
# 2016-10-31: We're experimentally expanding the environment ID format
# to include an optional username.
env_id_re: re.Pattern = re.compile(
    r"^(?:(?P<namespace>[\w:-]+)\/)?(?P<name>[\w:.-]+)-v(?P<version>\d+)$"
)


def env_id_from_parts(namespace: str, name: str, version: Union[int, str]) -> str:
    """
    Construct the environment ID from the namespace, name, and version.
    """
    namespace = "" if namespace is None else f"{namespace}/"
    return f"{namespace}{name}-v{version}"


# Whitelist of plugins which can hook into the `gym.envs.internal` entry point.
plugin_internal_whitelist: Set[str] = {"ale_py.gym"}

# The following is a map of environments which have been relocated
# to a different namespace. This map is important when reporting
# new versions of an environment outside of Gym.
# This map should be removed eventually once users
# are sufficiently aware of the environment relocations.
# The value of the mapping is (namespace, package,).
internal_env_namespace_relocation_map: Dict[str, Tuple[str, str]] = {
    "Adventure": (
        "ALE",
        "ale-py",
    ),
    "AirRaid": (
        "ALE",
        "ale-py",
    ),
    "Alien": (
        "ALE",
        "ale-py",
    ),
    "Amidar": (
        "ALE",
        "ale_py",
    ),
    "Assault": (
        "ALE",
        "ale-py",
    ),
    "Asterix": (
        "ALE",
        "ale-py",
    ),
    "Asteroids": (
        "ALE",
        "ale-py",
    ),
    "Atlantis": (
        "ALE",
        "ale-py",
    ),
    "BankHeist": (
        "ALE",
        "ale-py",
    ),
    "BattleZone": (
        "ALE",
        "ale-py",
    ),
    "BeamRider": (
        "ALE",
        "ale-py",
    ),
    "Berzerk": (
        "ALE",
        "ale-py",
    ),
    "Bowling": (
        "ALE",
        "ale-py",
    ),
    "Boxing": (
        "ALE",
        "ale-py",
    ),
    "Breakout": (
        "ALE",
        "ale-py",
    ),
    "Carnival": (
        "ALE",
        "ale-py",
    ),
    "Centipede": (
        "ALE",
        "ale-py",
    ),
    "ChopperCommand": (
        "ALE",
        "ale-py",
    ),
    "CrazyClimber": (
        "ALE",
        "ale-py",
    ),
    "Defender": (
        "ALE",
        "ale-py",
    ),
    "DemonAttack": (
        "ALE",
        "ale-py",
    ),
    "DoubleDunk": (
        "ALE",
        "ale-py",
    ),
    "ElevatorAction": (
        "ALE",
        "ale-py",
    ),
    "Enduro": (
        "ALE",
        "ale-py",
    ),
    "FishingDerby": (
        "ALE",
        "ale-py",
    ),
    "Freeway": (
        "ALE",
        "ale-py",
    ),
    "Frostbite": (
        "ALE",
        "ale-py",
    ),
    "Gopher": (
        "ALE",
        "ale-py",
    ),
    "Gravitar": (
        "ALE",
        "ale-py",
    ),
    "Hero": (
        "ALE",
        "ale-py",
    ),
    "IceHockey": (
        "ALE",
        "ale-py",
    ),
    "Jamesbond": (
        "ALE",
        "ale-py",
    ),
    "JourneyEscape": (
        "ALE",
        "ale-py",
    ),
    "Kangaroo": (
        "ALE",
        "ale-py",
    ),
    "Krull": (
        "ALE",
        "ale-py",
    ),
    "KungFuMaster": (
        "ALE",
        "ale-py",
    ),
    "MontezumaRevenge": (
        "ALE",
        "ale-py",
    ),
    "MsPacman": (
        "ALE",
        "ale-py",
    ),
    "NameThisGame": (
        "ALE",
        "ale-py",
    ),
    "Phoenix": (
        "ALE",
        "ale-py",
    ),
    "Pitfall": (
        "ALE",
        "ale-py",
    ),
    "Pong": (
        "ALE",
        "ale-py",
    ),
    "Pooyan": (
        "ALE",
        "ale-py",
    ),
    "PrivateEye": (
        "ALE",
        "ale-py",
    ),
    "Qbert": (
        "ALE",
        "ale-py",
    ),
    "Riverraid": (
        "ALE",
        "ale-py",
    ),
    "RoadRunner": (
        "ALE",
        "ale-py",
    ),
    "Robotank": (
        "ALE",
        "ale-py",
    ),
    "Seaquest": (
        "ALE",
        "ale-py",
    ),
    "Skiing": (
        "ALE",
        "ale-py",
    ),
    "Solaris": (
        "ALE",
        "ale-py",
    ),
    "SpaceInvaders": (
        "ALE",
        "ale-py",
    ),
    "StarGunner": (
        "ALE",
        "ale-py",
    ),
    "Tennis": (
        "ALE",
        "ale-py",
    ),
    "TimePilot": (
        "ALE",
        "ale-py",
    ),
    "Tutankham": (
        "ALE",
        "ale-py",
    ),
    "UpNDown": (
        "ALE",
        "ale-py",
    ),
    "Venture": (
        "ALE",
        "ale-py",
    ),
    "VideoPinball": (
        "ALE",
        "ale-py",
    ),
    "WizardOfWor": (
        "ALE",
        "ale-py",
    ),
    "YarsRevenge": (
        "ALE",
        "ale-py",
    ),
    "Zaxxon": (
        "ALE",
        "ale-py",
    ),
}


def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class EnvSpec:
    """A specification for a particular instance of the environment. Used
    to register the parameters for official evaluations.

    Args:
        id (str): The official environment ID
        entry_point (Optional[str]): The Python entrypoint of the environment class (e.g. module.name:Class)
        reward_threshold (Optional[int]): The reward threshold before the task is considered solved
        nondeterministic (bool): Whether this environment is non-deterministic even after seeding
        max_episode_steps (Optional[int]): The maximum number of steps that an episode can consist of
        order_enforce (Optional[int]): Whether to wrap the environment in an orderEnforcing wrapper
        kwargs (dict): The kwargs to pass to the environment class

    """

    def __init__(
        self,
        id,
        entry_point=None,
        reward_threshold=None,
        nondeterministic=False,
        max_episode_steps=None,
        order_enforce=True,
        kwargs=None,
    ):
        self.id = id
        self.entry_point = entry_point
        self.reward_threshold = reward_threshold
        self.nondeterministic = nondeterministic
        self.max_episode_steps = max_episode_steps
        self.order_enforce = order_enforce
        self._kwargs = {} if kwargs is None else kwargs

        match = env_id_re.fullmatch(id)
        if not match:
            raise error.Error(
                f"Attempted to register malformed environment ID: {id}."
                f"(Currently all IDs must be of the form {env_id_re.pattern}.)"
            )
        self._env_name = match.group("name")

    def make(self, **kwargs):
        """Instantiates an instance of the environment with appropriate kwargs"""
        if self.entry_point is None:
            raise error.Error(
                f"Attempting to make deprecated env {self.id}. "
                "(HINT: is there a newer registered version of this env?)"
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
        if env.spec.max_episode_steps is not None:
            from gym.wrappers.time_limit import TimeLimit

            env = TimeLimit(env, max_episode_steps=env.spec.max_episode_steps)
        else:
            if self.order_enforce:
                from gym.wrappers.order_enforcing import OrderEnforcing

                env = OrderEnforcing(env)
        return env

    def __repr__(self):
        return f"EnvSpec({self.id})"


class EnvSpecTree(MutableMapping):
    """
    The EnvSpecTree provides a dict-like mapping object
    from environment IDs to specifications.

    The EnvSpecTree is backed by a tree-like structure.
    The environment ID format is [{namespace}/]{name}-v{version}.

    The tree has multiple root nodes corresponding to a namespace.
    The children of a namespace node corresponds to the environment name.
    Furthermore, each name has a mapping from versions to specifications.
    It looks like the following,

    {
        None: {
            MountainCar: {
                0: EnvSpec(...),
                1: EnvSpec(...)
            }
        },
        ALE: {
            Tetris: {
                5: EnvSpec(...)
            }
        }
    }

    The tree-structure isn't user-facing and the EnvSpecTree will act
    like a dictionary. For example, to lookup an environment ID:

        ```
        specs = EnvSpecTree()

        specs["My/Env-v0"] = EnvSpec(...)
        assert specs["My/Env-v0"] == EnvSpec(...)

        assert specs.tree["My"]["Env"]["0"] == specs["My/Env-v0"]
        ```
    """

    def __init__(self):
        # Initialize the tree as a nested sequence of defaultdicts
        self.tree = defaultdict(lambda: defaultdict(dict))
        self._length = 0

    def __iter__(self) -> Generator[str, None, None]:
        # Iterate through the structure and generate the IDs contained
        # in the tree.
        for namespace, names in self.tree.items():
            for name, versions in names.items():
                for version in versions.keys():
                    yield env_id_from_parts(namespace, name, version)

    def _get_matches(self, key: str) -> Tuple[str, str, str]:
        # Match the regular expression against a full ID
        # to parse the associated namespace, name, and version.
        match = env_id_re.fullmatch(key)
        if match is None:
            raise KeyError(f"Malformed environment spec key {key}.")
        return match.group("namespace", "name", "version")

    def _exists(self, namespace: Optional[str], name: str, version: str) -> bool:
        # Helper which can look if an ID exists in the tree.
        if (
            namespace in self.tree
            and name in self.tree[namespace]
            and version in self.tree[namespace][name]
        ):
            return True
        else:
            return False

    def __getitem__(self, key: str) -> EnvSpec:
        # Get an item from the tree.
        # We first parse the components so we can look up the
        # appropriate environment ID.
        namespace, name, version = self._get_matches(key)
        if not self._exists(namespace, name, version):
            raise KeyError(f"{key}")

        return self.tree[namespace][name][version]

    def __setitem__(self, key: str, value: EnvSpec) -> None:
        # Insert an item into the tree.
        # First we parse the components to get the path
        # for insertion.
        namespace, name, version = self._get_matches(key)
        self.tree[namespace][name][version] = value
        # Increase the size
        self._length += 1

    def __delitem__(self, key: str) -> None:
        # Delete an item from the tree.
        # First parse the components so we can follow the
        # path to delete.
        namespace, name, version = self._get_matches(key)
        if not self._exists(namespace, name, version):
            raise KeyError(f"{key}")

        # Remove the envspec with this version.
        self.tree[namespace][name].pop(version)
        # Remove the name if it's empty.
        if len(self.tree[namespace][name]) == 0:
            self.tree[namespace].pop(name)
        # Remove the namespace if it's empty.
        if len(self.tree[namespace]) == 0:
            self.tree.pop(namespace)
        # Decrease the size
        self._length -= 1

    def __contains__(self, key: str) -> bool:
        # Check if the tree contains a path for this key.
        namespace, name, version = self._get_matches(key)
        return self._exists(namespace, name, version)

    def __repr__(self) -> str:
        # Construct a tree-like representation structure
        # so we can easily look at the contents of the tree.
        tree_repr = ""
        for namespace, names in self.tree.items():
            # For each namespace we'll iterate over the names
            root = namespace is None
            # Insert a separator if we're between depths
            if len(tree_repr) > 0:
                tree_repr += "│\n"
            # if this isn't the root we'll display the namespace
            if not root:
                tree_repr += f"├──{str(namespace)}\n"

            # Construct the namespace string so we can print this for
            # our children.
            namespace = f"{namespace}/" if namespace is not None else ""
            for name_idx, (name, versions) in enumerate(names.items()):
                # If this isn't the root we'll have to increase our
                # depth, i.e., insert some space
                if not root:
                    tree_repr += "│   "
                # If this is the last item make sure we use the
                # termination character. Otherwise use the nested
                # character.
                if name_idx == len(names) - 1:
                    tree_repr += "└──"
                else:
                    tree_repr += "├──"
                # Print the namespace and the name
                # and get ready to print the versions.
                tree_repr += f"{namespace}{name}: [ "
                # Print each version comma separated
                for version_idx, version in enumerate(versions.keys()):
                    tree_repr += f"v{version}"
                    if version_idx < len(versions) - 1:
                        tree_repr += ","
                    tree_repr += " "
                tree_repr += " ]\n"

        return tree_repr

    def __len__(self):
        # Return the length of the container
        return self._length


class EnvRegistry:
    """Register an env by ID. IDs remain stable over time and are
    guaranteed to resolve to the same environment dynamics (or be
    desupported). The goal is that results on a particular environment
    should always be comparable, and not depend on the version of the
    code that was running.
    """

    def __init__(self):
        self.env_specs = EnvSpecTree()
        self._ns = None

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            logger.info("Making new env: %s (%s)", path, kwargs)
        else:
            logger.info("Making new env: %s", path)
        spec = self.spec(path)

        # Match the parts of the environment ID as we need to parse
        # if there's a newer version of this environment.
        match = env_id_re.fullmatch(spec.id)
        assert match is not None  # Can't be hit as self.spec checks
        namespace, name, version = match.group("namespace", "name", "version")

        # Get all versions in the requested namespace.
        versions = self._versions(namespace, name)

        # We check what the latest version of the environment is and display a warning
        # if the user is attempting to initialize an older version.
        latest_version, latest_ns = max(versions)
        if int(version) < latest_version:
            latest_id = env_id_from_parts(latest_ns, name, latest_version)
            logger.warn(
                f"The environment {spec.id} is out of date. You should consider "
                f"upgrading to version v{latest_version} "
                f"with the environment ID `{latest_id}`."
            )

        env = spec.make(**kwargs)
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
                    f"A module ({mod_name}) was specified for the environment but was not found, "
                    "make sure the package is installed with `pip install` before calling `gym.make()`"
                )
        else:
            id = path

        match = env_id_re.fullmatch(id)
        if not match:
            raise error.Error(
                f"Attempted to look up malformed environment ID: {id.encode('utf-8')}. "
                f"(Currently all IDs must be of the form {env_id_re.pattern}.)"
            )

        namespace, name, version = match.group("namespace", "name", "version")

        # Check if namespace exists
        if namespace not in self.env_specs.tree:
            raise error.UnregisteredEnv(
                f"Namespace {namespace} does not exist, have you installed the proper package for {namespace}?"
            )
        # Check if name exists in namespace
        elif name not in self.env_specs.tree[namespace]:
            # If this name has been relocated we'll construct a useful error message.
            if name in internal_env_namespace_relocation_map:
                # Get the relocated namespace and corresponding package
                (
                    relocated_namespace,
                    relocated_package,
                ) = internal_env_namespace_relocation_map[name]

                name_not_found_error_msg = f"{relocated_namespace} environment {name} has been moved out of Gym to the package {relocated_package}."
                # If this namespace and name is registered we should instruct the user
                # to construct it under the new namespace.
                if (
                    relocated_namespace in self.env_specs.tree
                    and name in self.env_specs.tree[relocated_namespace]
                ):
                    name_not_found_error_msg += f"Please instantiate the new namespaced environment as {relocated_namespace}/{name}."
                # Otherwise we'll instruct them to install the package
                # and then instantiate under the new namespace.
                else:
                    name_not_found_error_msg += (
                        f"Please install the package via `pip install {relocated_package}` and then instantiate the environment "
                        f"as `{relocated_namespace}/{name}`"
                    )
            # If this hasn't been relocated we'll construct a generic error message
            else:
                name_not_found_error_msg = f"Environment {id} doesn't exist"
                if namespace is not None:
                    name_not_found_error_msg += f" in namespace {namespace}."
                else:
                    name_not_found_error_msg += "."
            # Throw the error
            raise error.UnregisteredEnv(name_not_found_error_msg)
        # We'll now check if the requested version exists
        elif version not in self.env_specs.tree[namespace][name]:
            version_not_found_error_msg = f"Environment version {version} for "
            if namespace is not None:
                version_not_found_error_msg += f"{namespace}/"
            version_not_found_error_msg += (
                f"{name} could not be found. Valid versions are: "
            )
            # Retrieve valid versions of this package and print them
            versions = self._versions(namespace, name)
            version_not_found_error_msg += ", ".join(
                map(
                    lambda version: env_id_from_parts(
                        getitem(version, 1), name, getitem(version, 0)
                    ),
                    versions,
                )
            )
            version_not_found_error_msg += "."

            # If we've requested a version less than the
            # most recent version it's considered deprecated.
            # Otherwise it isn't registered.
            if int(version) < getitem(max(versions), 0):
                raise error.DeprecatedEnv(version_not_found_error_msg)
            else:
                raise error.UnregisteredEnv(version_not_found_error_msg)

        return self.env_specs[id]

    def register(self, id, **kwargs):
        # Match ID and and get environment parts
        match = env_id_re.fullmatch(id)
        if match is None:
            raise error.Error(
                f"Attempted to register malformed environment ID: {id.encode('utf-8')}. "
                f"(Currently all IDs must be of the form {env_id_re.pattern}.)"
            )

        if self._ns is not None:
            namespace, name, version = match.group("namespace", "name", "version")
            if namespace is not None:
                logger.warn(
                    f"Custom namespace '{namespace}' is being overridden by namespace '{self._ns}'. "
                    "If you are developing a plugin you shouldn't specify a namespace in `register` calls. "
                    "The namespace is specified through the entry point key."
                )
            # Replace namespace
            id = env_id_from_parts(self._ns, name, version)

        if id in self.env_specs:
            logger.warn(f"Overriding environment {id}")
        self.env_specs[id] = EnvSpec(id, **kwargs)

    def _versions(self, namespace: str, name: str) -> Set[Tuple[int, str]]:
        # Get the set of versions under the requested namespace
        versions = set(
            map(
                lambda version: (
                    int(version),
                    namespace,
                ),
                self.env_specs.tree[namespace][name].keys(),
            )
        )

        # The newest environment may be outside of the current namespace.
        # This happens for internal environments which have been factored out of Gym.
        # We check if the name has been relocated and we attempt to add the new
        # versions outside of Gym to our set.
        if name in internal_env_namespace_relocation_map:
            relocated_namespace, _ = internal_env_namespace_relocation_map[name]

            # If this name exists under the new namespace
            # we'll add these versions to the set.
            if name in self.env_specs.tree[relocated_namespace]:
                versions |= set(
                    map(
                        lambda version: (
                            int(version),
                            relocated_namespace,
                        ),
                        self.env_specs.tree[relocated_namespace][name].keys(),
                    )
                )

        return versions

    @contextlib.contextmanager
    def namespace(self, ns):
        self._ns = ns
        yield
        self._ns = None


# Have a global registry
registry = EnvRegistry()


def register(id, **kwargs):
    return registry.register(id, **kwargs)


def make(id, **kwargs):
    return registry.make(id, **kwargs)


def spec(id):
    return registry.spec(id)


@contextlib.contextmanager
def namespace(ns):
    with registry.namespace(ns):
        yield


def load_env_plugins(entry_point="gym.envs"):
    # Load third-party environments
    for plugin in metadata.entry_points().get(entry_point, []):
        # Python 3.8 doesn't support plugin.module, plugin.attr
        # So we'll have to try and parse this ourselves
        try:
            module, attr = plugin.module, plugin.attr
        except AttributeError:
            if ":" in plugin.value:
                module, attr = plugin.value.split(":", maxsplit=1)
            else:
                module, attr = plugin.value, None
        finally:
            if attr is None:
                raise error.Error(
                    f"Gym environment plugin `{module}` must specify a function to execute, not a root module"
                )

        context = namespace(plugin.name)
        if plugin.name == "__internal__":
            if module in plugin_internal_whitelist:
                context = contextlib.nullcontext()
            else:
                logger.warn(
                    f"Trying to register an internal environment when `{module}` is not in the whitelist"
                )

        with context:
            fn = plugin.load()
            try:
                fn()
            except Exception as e:
                logger.warn(str(e))
