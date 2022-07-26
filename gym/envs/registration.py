import contextlib
import copy
import difflib
import importlib
import importlib.util
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    SupportsFloat,
    Tuple,
    Union,
    overload,
)

import numpy as np

from gym.envs.__relocated__ import internal_env_relocation_map
from gym.wrappers import (
    AutoResetWrapper,
    HumanRendering,
    OrderEnforcing,
    StepAPICompatibility,
    TimeLimit,
)
from gym.wrappers.env_checker import PassiveEnvChecker

if sys.version_info < (3, 10):
    import importlib_metadata as metadata  # type: ignore
else:
    import importlib.metadata as metadata

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from gym import Env, error, logger

ENV_ID_RE = re.compile(
    r"^(?:(?P<namespace>[\w:-]+)\/)?(?:(?P<name>[\w:.-]+?))(?:-v(?P<version>\d+))?$"
)


def load(name: str) -> callable:
    """Loads an environment with name and returns an environment creation function

    Args:
        name: The environment name

    Returns:
        Calls the environment constructor
    """
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def parse_env_id(id: str) -> Tuple[Optional[str], str, Optional[int]]:
    """Parse environment ID string format.

    This format is true today, but it's *not* an official spec.
    [namespace/](env-name)-v(version)    env-name is group 1, version is group 2

    2016-10-31: We're experimentally expanding the environment ID format
    to include an optional namespace.

    Args:
        id: The environment id to parse

    Returns:
        A tuple of environment namespace, environment name and version number

    Raises:
        Error: If the environment id does not a valid environment regex
    """
    match = ENV_ID_RE.fullmatch(id)
    if not match:
        raise error.Error(
            f"Malformed environment ID: {id}."
            f"(Currently all IDs must be of the form [namespace/](env-name)-v(version). (namespace is optional))"
        )
    namespace, name, version = match.group("namespace", "name", "version")
    if version is not None:
        version = int(version)

    return namespace, name, version


def get_env_id(ns: Optional[str], name: str, version: Optional[int]) -> str:
    """Get the full env ID given a name and (optional) version and namespace. Inverse of :meth:`parse_env_id`.

    Args:
        ns: The environment namespace
        name: The environment name
        version: The environment version

    Returns:
        The environment id
    """

    full_name = name
    if version is not None:
        full_name += f"-v{version}"
    if ns is not None:
        full_name = ns + "/" + full_name
    return full_name


@dataclass
class EnvSpec:
    """A specification for creating environments with `gym.make`.

    * id: The string used to create the environment with `gym.make`
    * entry_point: The location of the environment to create from
    * reward_threshold: The reward threshold for completing the environment.
    * nondeterministic: If the observation of an environment cannot be repeated with the same initial state, random number generator state and actions.
    * max_episode_steps: The max number of steps that the environment can take before truncation
    * order_enforce: If to enforce the order of `reset` before `step` and `render` functions
    * autoreset: If to automatically reset the environment on episode end
    * disable_env_checker: If to disable the environment checker wrapper in `gym.make`, by default False (runs the environment checker)
    * kwargs: Additional keyword arguments passed to the environments through `gym.make`
    """

    id: str
    entry_point: Optional[Union[Callable, str]] = field(default=None)
    reward_threshold: Optional[float] = field(default=None)
    nondeterministic: bool = field(default=False)

    # Wrappers
    max_episode_steps: Optional[int] = field(default=None)
    order_enforce: bool = field(default=True)
    autoreset: bool = field(default=False)
    disable_env_checker: bool = field(default=False)
    new_step_api: bool = field(default=False)

    # Environment arguments
    kwargs: dict = field(default_factory=dict)

    namespace: Optional[str] = field(init=False)
    name: str = field(init=False)
    version: Optional[int] = field(init=False)

    def __post_init__(self):
        # Initialize namespace, name, version
        self.namespace, self.name, self.version = parse_env_id(self.id)

    def make(self, **kwargs) -> Env:
        # For compatibility purposes
        return make(self, **kwargs)


def _check_namespace_exists(ns: Optional[str]):
    """Check if a namespace exists. If it doesn't, print a helpful error message."""
    if ns is None:
        return
    namespaces = {
        spec_.namespace for spec_ in registry.values() if spec_.namespace is not None
    }
    if ns in namespaces:
        return

    suggestion = (
        difflib.get_close_matches(ns, namespaces, n=1) if len(namespaces) > 0 else None
    )
    suggestion_msg = (
        f"Did you mean: `{suggestion[0]}`?"
        if suggestion
        else f"Have you installed the proper package for {ns}?"
    )

    raise error.NamespaceNotFound(f"Namespace {ns} not found. {suggestion_msg}")


def _check_name_exists(ns: Optional[str], name: str):
    """Check if an env exists in a namespace. If it doesn't, print a helpful error message."""
    _check_namespace_exists(ns)
    names = {spec_.name for spec_ in registry.values() if spec_.namespace == ns}

    if name in names:
        return

    if namespace is None and name in internal_env_relocation_map:
        relocated_namespace, relocated_package = internal_env_relocation_map[name]
        message = f"The environment `{name}` has been moved out of Gym to the package `{relocated_package}`."

        # Check if the package is installed
        # If not instruct the user to install the package and then how to instantiate the env
        if importlib.util.find_spec(relocated_package) is None:
            message += (
                f" Please install the package via `pip install {relocated_package}`."
            )

        # Otherwise the user should be able to instantiate the environment directly
        if namespace != relocated_namespace:
            message += f" You can instantiate the new namespaced environment as `{relocated_namespace}/{name}`."

        raise error.NameNotFound(message)

    suggestion = difflib.get_close_matches(name, names, n=1)
    namespace_msg = f" in namespace {ns}" if ns else ""
    suggestion_msg = f"Did you mean: `{suggestion[0]}`?" if suggestion else ""

    raise error.NameNotFound(
        f"Environment {name} doesn't exist{namespace_msg}. {suggestion_msg}"
    )


def _check_version_exists(ns: Optional[str], name: str, version: Optional[int]):
    """Check if an env version exists in a namespace. If it doesn't, print a helpful error message.
    This is a complete test whether an environment identifier is valid, and will provide the best available hints.

    Args:
        ns: The environment namespace
        name: The environment space
        version: The environment version

    Raises:
        DeprecatedEnv: The environment doesn't exist but a default version does
        VersionNotFound: The ``version`` used doesn't exist
        DeprecatedEnv: Environment version is deprecated
    """
    if get_env_id(ns, name, version) in registry:
        return

    _check_name_exists(ns, name)
    if version is None:
        return

    message = f"Environment version `v{version}` for environment `{get_env_id(ns, name, None)}` doesn't exist."

    env_specs = [
        spec_
        for spec_ in registry.values()
        if spec_.namespace == ns and spec_.name == name
    ]
    env_specs = sorted(env_specs, key=lambda spec_: int(spec_.version or -1))

    default_spec = [spec_ for spec_ in env_specs if spec_.version is None]

    if default_spec:
        message += f" It provides the default version {default_spec[0].id}`."
        if len(env_specs) == 1:
            raise error.DeprecatedEnv(message)

    # Process possible versioned environments

    versioned_specs = [spec_ for spec_ in env_specs if spec_.version is not None]

    latest_spec = max(versioned_specs, key=lambda spec: spec.version, default=None)  # type: ignore
    if latest_spec is not None and version > latest_spec.version:
        version_list_msg = ", ".join(f"`v{spec_.version}`" for spec_ in env_specs)
        message += f" It provides versioned environments: [ {version_list_msg} ]."

        raise error.VersionNotFound(message)

    if latest_spec is not None and version < latest_spec.version:
        raise error.DeprecatedEnv(
            f"Environment version v{version} for `{get_env_id(ns, name, None)}` is deprecated. "
            f"Please use `{latest_spec.id}` instead."
        )


def find_highest_version(ns: Optional[str], name: str) -> Optional[int]:
    version: List[int] = [
        spec_.version
        for spec_ in registry.values()
        if spec_.namespace == ns and spec_.name == name and spec_.version is not None
    ]
    return max(version, default=None)


def load_env_plugins(entry_point: str = "gym.envs") -> None:
    # Load third-party environments
    for plugin in metadata.entry_points(group=entry_point):
        # Python 3.8 doesn't support plugin.module, plugin.attr
        # So we'll have to try and parse this ourselves
        module, attr = None, None
        try:
            module, attr = plugin.module, plugin.attr  # type: ignore  ## error: Cannot access member "attr" for type "EntryPoint"
        except AttributeError:
            if ":" in plugin.value:
                module, attr = plugin.value.split(":", maxsplit=1)
            else:
                module, attr = plugin.value, None
        except Exception as e:
            warnings.warn(
                f"While trying to load plugin `{plugin}` from {entry_point}, an exception occurred: {e}"
            )
            module, attr = None, None
        finally:
            if attr is None:
                raise error.Error(
                    f"Gym environment plugin `{module}` must specify a function to execute, not a root module"
                )

        context = namespace(plugin.name)
        if plugin.name.startswith("__") and plugin.name.endswith("__"):
            # `__internal__` is an artifact of the plugin system when
            # the root namespace had an allow-list. The allow-list is now
            # removed and plugins can register environments in the root
            # namespace with the `__root__` magic key.
            if plugin.name == "__root__" or plugin.name == "__internal__":
                context = contextlib.nullcontext()
            else:
                logger.warn(
                    f"The environment namespace magic key `{plugin.name}` is unsupported. "
                    "To register an environment at the root namespace you should specify "
                    "the `__root__` namespace."
                )

        with context:
            fn = plugin.load()
            try:
                fn()
            except Exception as e:
                logger.warn(str(e))


# fmt: off
# Classic control
# ----------------------------------------


@overload
def make(id: Literal["CartPole-v0", "CartPole-v1"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, int]]: ...
@overload
def make(id: Literal["MountainCar-v0"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, int]]: ...
@overload
def make(id: Literal["MountainCarContinuous-v0"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, Sequence[SupportsFloat]]]: ...
@overload
def make(id: Literal["Pendulum-v1"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, Sequence[SupportsFloat]]]: ...
@overload
def make(id: Literal["Acrobot-v1"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, int]]: ...

# Box2d
# ----------------------------------------


@overload
def make(id: Literal["LunarLander-v2", "LunarLanderContinuous-v2"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, int]]: ...
@overload
def make(id: Literal["BipedalWalker-v3", "BipedalWalkerHardcore-v3"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, Sequence[SupportsFloat]]]: ...
@overload
def make(id: Literal["CarRacing-v1"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, Sequence[SupportsFloat]]]: ...

# Toy Text
# ----------------------------------------


@overload
def make(id: Literal["Blackjack-v1"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, int]]: ...
@overload
def make(id: Literal["FrozenLake-v1", "FrozenLake8x8-v1"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, int]]: ...
@overload
def make(id: Literal["CliffWalking-v0"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, int]]: ...
@overload
def make(id: Literal["Taxi-v3"], **kwargs) -> Env[np.ndarray, Union[np.ndarray, int]]: ...

# Mujoco
# ----------------------------------------


@overload
def make(id: Literal[
    "Reacher-v2",
    "Pusher-v2",
    "Thrower-v2",
    "Striker-v2",
    "InvertedPendulum-v2",
    "InvertedDoublePendulum-v2",
    "HalfCheetah-v2", "HalfCheetah-v3",
    "Hopper-v2", "Hopper-v3",
    "Swimmer-v2", "Swimmer-v3",
    "Walker2d-v2", "Walker2d-v3",
    "Ant-v2"
], **kwargs) -> Env[np.ndarray, np.ndarray]: ...


@overload
def make(id: str, **kwargs) -> Env: ...
@overload
def make(id: EnvSpec, **kwargs) -> Env: ...

# fmt: on


class EnvRegistry(dict):
    """A glorified dictionary for compatibility reasons.

    Turns out that some existing code directly used the old `EnvRegistry` code,
    even though the intended API was just `register` and `make`.
    This reimplements some old methods, so that e.g. pybullet environments will still work.

    Ideally, nobody should ever use these methods, and they will be removed soon.
    """

    # TODO: remove this at 1.0

    def make(self, path: str, **kwargs) -> Env:
        logger.warn(
            "The `registry.make` method is deprecated. Please use `gym.make` instead."
        )
        return make(path, **kwargs)

    def register(self, id: str, **kwargs) -> None:
        logger.warn(
            "The `registry.register` method is deprecated. Please use `gym.register` instead."
        )
        return register(id, **kwargs)

    def all(self) -> Iterable[EnvSpec]:
        logger.warn(
            "The `registry.all` method is deprecated. Please use `registry.values` instead."
        )
        return self.values()

    def spec(self, path: str) -> EnvSpec:
        logger.warn(
            "The `registry.spec` method is deprecated. Please use `gym.spec` instead."
        )
        return spec(path)

    def namespace(self, ns: str):
        logger.warn(
            "The `registry.namespace` method is deprecated. Please use `gym.namespace` instead."
        )
        return namespace(ns)

    @property
    def env_specs(self):
        logger.warn(
            "The `registry.env_specs` property along with `EnvSpecTree` is deprecated. Please use `registry` directly as a dictionary instead."
        )
        return self


# Global registry of environments. Meant to be accessed through `register` and `make`
registry: Dict[str, EnvSpec] = EnvRegistry()
current_namespace: Optional[str] = None


def _check_spec_register(spec: EnvSpec):
    """Checks whether the spec is valid to be registered. Helper function for `register`."""
    global registry
    latest_versioned_spec = max(
        (
            spec_
            for spec_ in registry.values()
            if spec_.namespace == spec.namespace
            and spec_.name == spec.name
            and spec_.version is not None
        ),
        key=lambda spec_: int(spec_.version),  # type: ignore
        default=None,
    )

    unversioned_spec = next(
        (
            spec_
            for spec_ in registry.values()
            if spec_.namespace == spec.namespace
            and spec_.name == spec.name
            and spec_.version is None
        ),
        None,
    )

    if unversioned_spec is not None and spec.version is not None:
        raise error.RegistrationError(
            "Can't register the versioned environment "
            f"`{spec.id}` when the unversioned environment "
            f"`{unversioned_spec.id}` of the same name already exists."
        )
    elif latest_versioned_spec is not None and spec.version is None:
        raise error.RegistrationError(
            "Can't register the unversioned environment "
            f"`{spec.id}` when the versioned environment "
            f"`{latest_versioned_spec.id}` of the same name "
            f"already exists. Note: the default behavior is "
            f"that `gym.make` with the unversioned environment "
            f"will return the latest versioned environment"
        )


# Public API


@contextlib.contextmanager
def namespace(ns: str):
    global current_namespace
    old_namespace = current_namespace
    current_namespace = ns
    yield
    current_namespace = old_namespace


def register(id: str, **kwargs):
    """Register an environment with gym.

    The `id` parameter corresponds to the name of the environment, with the syntax as follows:
    `(namespace)/(env_name)-v(version)` where `namespace` is optional.

    It takes arbitrary keyword arguments, which are passed to the `EnvSpec` constructor.

    Args:
        id: The environment id
        **kwargs: arbitrary keyword arguments which are passed to the environment constructor
    """
    global registry, current_namespace
    ns, name, version = parse_env_id(id)

    if current_namespace is not None:
        if (
            kwargs.get("namespace") is not None
            and kwargs.get("namespace") != current_namespace
        ):
            logger.warn(
                f"Custom namespace `{kwargs.get('namespace')}` is being overridden "
                f"by namespace `{current_namespace}`. If you are developing a "
                "plugin you shouldn't specify a namespace in `register` "
                "calls. The namespace is specified through the "
                "entry point package metadata."
            )
        ns_id = current_namespace
    else:
        ns_id = ns

    full_id = get_env_id(ns_id, name, version)

    spec = EnvSpec(id=full_id, **kwargs)
    _check_spec_register(spec)
    if spec.id in registry:
        logger.warn(f"Overriding environment {spec.id}")
    registry[spec.id] = spec


def make(
    id: Union[str, EnvSpec],
    max_episode_steps: Optional[int] = None,
    autoreset: bool = False,
    new_step_api: bool = False,
    disable_env_checker: Optional[bool] = None,
    **kwargs,
) -> Env:
    """Create an environment according to the given ID.

    Args:
        id: Name of the environment. Optionally, a module to import can be included, eg. 'module:Env-v0'
        max_episode_steps: Maximum length of an episode (TimeLimit wrapper).
        autoreset: Whether to automatically reset the environment after each episode (AutoResetWrapper).
        new_step_api: Whether to use old or new step API (StepAPICompatibility wrapper). Will be removed at v1.0
        disable_env_checker: If to run the env checker, None will default to the environment specification `disable_env_checker`
            (which is by default False, running the environment checker),
            otherwise will run according to this parameter (`True` = not run, `False` = run)
        kwargs: Additional arguments to pass to the environment constructor.

    Returns:
        An instance of the environment.

    Raises:
        Error: If the ``id`` doesn't exist then an error is raised
    """
    if isinstance(id, EnvSpec):
        spec_ = id
    else:
        module, id = (None, id) if ":" not in id else id.split(":")
        if module is not None:
            try:
                importlib.import_module(module)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"{e}. Environment registration via importing a module failed. "
                    f"Check whether '{module}' contains env registration and can be imported."
                )
        spec_ = registry.get(id)

        ns, name, version = parse_env_id(id)
        latest_version = find_highest_version(ns, name)
        if (
            version is not None
            and latest_version is not None
            and latest_version > version
        ):
            logger.warn(
                f"The environment {id} is out of date. You should consider "
                f"upgrading to version `v{latest_version}`."
            )
        if version is None and latest_version is not None:
            version = latest_version
            new_env_id = get_env_id(ns, name, version)
            spec_ = registry.get(new_env_id)
            logger.warn(
                f"Using the latest versioned environment `{new_env_id}` "
                f"instead of the unversioned environment `{id}`."
            )

        if spec_ is None:
            _check_version_exists(ns, name, version)
            raise error.Error(f"No registered env with id: {id}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if spec_.entry_point is None:
        raise error.Error(f"{spec_.id} registered but entry_point is not specified")
    elif callable(spec_.entry_point):
        env_creator = spec_.entry_point
    else:
        # Assume it's a string
        env_creator = load(spec_.entry_point)

    mode = _kwargs.get("render_mode")
    apply_human_rendering = False

    # If we have access to metadata we check that "render_mode" is valid and see if the HumanRendering wrapper needs to be applied
    if mode is not None and hasattr(env_creator, "metadata"):
        assert isinstance(
            env_creator.metadata, dict
        ), f"Expect the environment creator ({env_creator}) metadata to be dict, actual type: {type(env_creator.metadata)}"

        if "render_modes" in env_creator.metadata:
            render_modes = env_creator.metadata["render_modes"]
            if not isinstance(render_modes, Sequence):
                logger.warn(
                    f"Expects the environment metadata render_modes to be a Sequence (tuple or list), actual type: {type(render_modes)}"
                )

            # Apply the `HumanRendering` wrapper, if the mode=="human" but "human" not in render_modes
            if (
                mode == "human"
                and "human" not in render_modes
                and ("single_rgb_array" in render_modes or "rgb_array" in render_modes)
            ):
                logger.warn(
                    "You are trying to use 'human' rendering for an environment that doesn't natively support it. "
                    "The HumanRendering wrapper is being applied to your environment."
                )
                apply_human_rendering = True
                if "single_rgb_array" in render_modes:
                    _kwargs["render_mode"] = "single_rgb_array"
                else:
                    _kwargs["render_mode"] = "rgb_array"
            elif mode not in render_modes:
                logger.warn(
                    f"The environment is being initialised with mode ({mode}) that is not in the possible render_modes ({render_modes})."
                )
        else:
            logger.warn(
                f"The environment creator metadata doesn't include `render_modes`, contains: {list(env_creator.metadata.keys())}"
            )

    try:
        env = env_creator(**_kwargs)
    except TypeError as e:
        if (
            str(e).find("got an unexpected keyword argument 'render_mode'") >= 0
            and apply_human_rendering
        ):
            raise error.Error(
                f"You passed render_mode='human' although {id} doesn't implement human-rendering natively. "
                "Gym tried to apply the HumanRendering wrapper but it looks like your environment is using the old "
                "rendering API, which is not supported by the HumanRendering wrapper."
            )
        else:
            raise e

    # Copies the environment creation specification and kwargs to add to the environment specification details
    spec_ = copy.deepcopy(spec_)
    spec_.kwargs = _kwargs
    env.unwrapped.spec = spec_

    # Run the environment checker as the lowest level wrapper
    if disable_env_checker is False or (
        disable_env_checker is None and spec_.disable_env_checker is False
    ):
        env = PassiveEnvChecker(env)

    env = StepAPICompatibility(env, new_step_api)

    # Add the order enforcing wrapper
    if spec_.order_enforce:
        env = OrderEnforcing(env)

    # Add the time limit wrapper
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps, new_step_api)
    elif spec_.max_episode_steps is not None:
        env = TimeLimit(env, spec_.max_episode_steps, new_step_api)

    # Add the autoreset wrapper
    if autoreset:
        env = AutoResetWrapper(env, new_step_api)

    # Add human rendering wrapper
    if apply_human_rendering:
        env = HumanRendering(env)

    return env


def spec(env_id: str) -> EnvSpec:
    """Retrieve the spec for the given environment from the global registry."""
    spec_ = registry.get(env_id)
    if spec_ is None:
        ns, name, version = parse_env_id(env_id)
        _check_version_exists(ns, name, version)
        raise error.Error(f"No registered env with id: {env_id}")
    else:
        assert isinstance(spec_, EnvSpec)
        return spec_
