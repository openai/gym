"""Core API for Environment, Wrapper, ActionWrapper and ObservationWrapper."""
from __future__ import annotations

from abc import abstractmethod
from typing import Generic, Optional, SupportsFloat, TypeVar, Union

from gym import spaces
from gym.logger import deprecation
from gym.utils import seeding
from gym.utils.seeding import RandomNumberGenerator

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Env(Generic[ObsType, ActType]):
    """The main OpenAI Gym class.

    It encapsulates an environment with arbitrary behind-the-scenes dynamics.
    An environment can be partially or fully observed.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

    The methods are accessed publicly as "step", "reset", etc...
    """

    # Set this in SOME subclasses
    metadata = {"render_modes": []}
    reward_range = (-float("inf"), float("inf"))
    spec = None

    # Set these in ALL subclasses
    action_space: spaces.Space[ActType]
    observation_space: spaces.Space[ObsType]

    # Created
    _np_random: RandomNumberGenerator | None = None

    @property
    def np_random(self) -> RandomNumberGenerator:
        """Initializes the np_random field if not done already."""
        if self._np_random is None:
            self._np_random, seed = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: RandomNumberGenerator):
        self._np_random = value

    @abstractmethod
    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        This method returns a tuple ``(observation, reward, done, info)``

        Returns:
            observation (object): agent's observation of the current environment. This will be an element of the environment's :attr:`observation_space`. This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further :meth:`step` calls will return undefined results. A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully, a certain timelimit was exceeded, or the physics simulation has entered an invalid state. ``info`` may contain additional information regarding the reason for a ``done`` signal.
            info (dict): contains auxiliary diagnostic information (helpful for debugging, learning, and logging). This might, for instance, contain:

                - metrics that describe the agent's performance or
                - state variables that are hidden from observations or
                - information that distinguishes truncation and termination or
                - individual reward terms that are combined to produce the total reward
        """
        raise NotImplementedError

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        """Resets the environment to an initial state and returns an initial observation.

        This method should also reset the environment's random number
        generator(s) if ``seed`` is an integer or if the environment has not
        yet initialized a random number generator. If the environment already
        has a random number generator and :meth:`reset` is called with ``seed=None``,
        the RNG should not be reset.
        Moreover, :meth:`reset` should (in the typical use case) be called with an
        integer seed right after initialization and then never again.

        Args:
            seed (int or None):
                The seed that is used to initialize the environment's PRNG. If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed, a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset. If you pass an integer, the PRNG will be reset even if it already exists. Usually, you want to pass an integer *right after the environment has been initialized and then never again*. Please refer to the minimal example above to see this paradigm in action.
            return_info (bool): If true, return additional information along with initial observation. This info should be analogous to the info returned in :meth:`step`
            options (dict or None): Additional information to specify how the environment is reset (optional, depending on the specific environment)


        Returns:
            observation (object): Observation of the initial state. This will be an element of :attr:`observation_space` (usually a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (optional dictionary): This will *only* be returned if ``return_info=True`` is passed. It contains auxiliary information complementing ``observation``. This dictionary should be analogous to the ``info`` returned by :meth:`step`.
        """
        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    @abstractmethod
    def render(self, mode="human"):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        third-party environments may not support rendering at all.)
        By convention, if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render_modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example::

            class MyEnv(Env):
                metadata = {'render_modes': ['human', 'rgb_array']}

                def render(self, mode='human'):
                    if mode == 'rgb_array':
                        return np.array(...) # return RGB frame suitable for video
                    elif mode == 'human':
                        ... # pop up a window and render
                    else:
                        super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        deprecation(
            "Function `env.seed(seed)` is marked as deprecated and will be removed in the future. "
            "Please use `env.reset(seed=seed) instead."
        )
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def unwrapped(self) -> Env:
        """Completely unwrap this env.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        """Returns a string for the environment with the spec id if specified."""
        if self.spec is None:
            return f"<{type(self).__name__} instance>"
        else:
            return f"<{type(self).__name__}<{self.spec.id}>>"

    def __enter__(self):
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args):
        """Support with-statement for the environment."""
        self.close()
        # propagate exception
        return False


class Wrapper(Env[ObsType, ActType]):
    """Wraps the environment to allow a modular transformation.

    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code.

    .. note::

        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.

    """

    def __init__(self, env: Env):
        """Wraps the environment to allow a modular transformation."""
        self.env = env

        self._action_space: spaces.Space | None = None
        self._observation_space: spaces.Space | None = None
        self._reward_range: tuple[SupportsFloat, SupportsFloat] | None = None
        self._metadata: dict | None = None

    def __getattr__(self, name):
        """Gets an attribute if the name does not start with _."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)

    @property
    def spec(self):
        """Returns the environment spec."""
        return self.env.spec

    @classmethod
    def class_name(cls):
        """Returns the class name."""
        return cls.__name__

    @property
    def action_space(self) -> spaces.Space[ActType]:
        """Returns the action space."""
        if self._action_space is None:
            return self.env.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space: spaces.Space):
        """Sets the action space."""
        self._action_space = space

    @property
    def observation_space(self) -> spaces.Space:
        """Returns the observation space."""
        if self._observation_space is None:
            return self.env.observation_space
        return self._observation_space

    @observation_space.setter
    def observation_space(self, space: spaces.Space):
        """Sets the observation shape."""
        self._observation_space = space

    @property
    def reward_range(self) -> tuple[SupportsFloat, SupportsFloat]:
        """Return the reward range."""
        if self._reward_range is None:
            return self.env.reward_range
        return self._reward_range

    @reward_range.setter
    def reward_range(self, value: tuple[SupportsFloat, SupportsFloat]):
        """Sets the reward range."""
        self._reward_range = value

    @property
    def metadata(self) -> dict:
        """Returns the environment metadata."""
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        """Sets the metadata."""
        self._metadata = value

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        """Steps through the environment with action."""
        return self.env.step(action)

    def reset(self, **kwargs) -> Union[ObsType, tuple[ObsType, dict]]:
        """Resets the environment with kwargs."""
        return self.env.reset(**kwargs)

    def render(self, **kwargs):
        """Renders the environment with kwargs."""
        return self.env.render(**kwargs)

    def close(self):
        """Closes the environment."""
        return self.env.close()

    def seed(self, seed=None):
        """Seeds the environment."""
        return self.env.seed(seed)

    def __str__(self):
        """Returns the wrapper name and the unwrapped environment string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns string of the wrapper."""
        return str(self)

    @property
    def unwrapped(self) -> Env:
        """Unwraps all wrappers to the environment."""
        return self.env.unwrapped


class ObservationWrapper(Wrapper):
    """A wrapper that can modify the returning observations from a step."""

    def reset(self, **kwargs):
        """Resets the environment with the modified observations."""
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
            return self.observation(obs), info
        else:
            return self.observation(self.env.reset(**kwargs))

    def step(self, action):
        """Modifies the observation before returning the step."""
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    @abstractmethod
    def observation(self, observation):
        """Modifies the observations."""
        raise NotImplementedError


class RewardWrapper(Wrapper):
    """A wrapper that can modify the returning reward from a step."""

    def step(self, action):
        """Modifies the reward after the environment step."""
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

    @abstractmethod
    def reward(self, reward):
        """Modifies the reward."""
        raise NotImplementedError


class ActionWrapper(Wrapper):
    """A wrapper that can modify the action before stepping."""

    def step(self, action):
        """Runs the environment step using the modified action."""
        return self.env.step(self.action(action))

    @abstractmethod
    def action(self, action):
        """Modifies the action during the action."""
        raise NotImplementedError

    @abstractmethod
    def reverse_action(self, action):
        """Reverse an action, unused in project."""
        raise NotImplementedError
