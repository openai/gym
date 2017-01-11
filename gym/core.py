import logging
logger = logging.getLogger(__name__)

import numpy as np

from gym import error
from gym.utils import closer, reraise

env_closer = closer.Closer()

# Env-related abstractions

class Env(object):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        configure
        seed

    When implementing an environment, override the following methods
    in your subclass:

        _step
        _reset
        _render
        _close
        _configure
        _seed

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """

    def __new__(cls, *args, **kwargs):
        # We use __new__ since we want the env author to be able to
        # override __init__ without remembering to call super.
        env = super(Env, cls).__new__(cls)
        env._env_closer_id = env_closer.register(env)
        env._closed = False
        env._configured = False
        env._unwrapped = None

        # Will be automatically set when creating an environment via 'make'
        env.spec = None
        return env

    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-np.inf, np.inf)

    # Override in SOME subclasses
    def _close(self):
        pass

    def _configure(self):
        pass

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    # Override in ALL subclasses
    def _step(self, action): raise NotImplementedError
    def _reset(self): raise NotImplementedError
    def _render(self, mode='human', close=False):
        if close:
            return
        raise NotImplementedError
    def _seed(self, seed=None): return []

    # Do not override
    _owns_render = True

    @property
    def monitor(self):
        raise error.Error("env.monitor has been deprecated as of 12/23/2016. Remove your call to `env.monitor.start(directory)` and instead wrap your env with `env = gym.wrappers.Monitor(env, directory)` to record data.")

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        observation, reward, done, info = self._step(action)
        return observation, reward, done, info

    def reset(self):
        """Resets the state of the environment and returns an initial
        observation. Will call 'configure()' if not already called.

        Returns: observation (object): the initial observation of the
            space. (Initial reward is assumed to be 0.)
        """
        if self.metadata.get('configure.required') and not self._configured:
            logger.warning("Called reset on %s before configuring. Configuring automatically with default arguments", self)
            self.configure()
        observation = self._reset()
        return observation

    def render(self, mode='human', close=False):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        if close:
            return self._render(close=close)

        # This code can be useful for calling super() in a subclass.
        modes = self.metadata.get('render.modes', [])
        if len(modes) == 0:
            raise error.UnsupportedMode('{} does not support rendering (requested mode: {})'.format(self, mode))
        elif mode not in modes:
            raise error.UnsupportedMode('Unsupported rendering mode: {}. (Supported modes for {}: {})'.format(mode, self, modes))

        return self._render(mode=mode, close=close)

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        # _closed will be missing if this instance is still
        # initializing.
        if not hasattr(self, '_closed') or self._closed:
            return

        if self._owns_render:
            self.render(close=True)

        self._close()
        env_closer.unregister(self._env_closer_id)
        # If an error occurs before this line, it's possible to
        # end up with double close.
        self._closed = True

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
        return self._seed(seed)

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.

        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        self._configured = True

        try:
            self._configure(*args, **kwargs)
        except TypeError as e:
            # It can be confusing if you have the wrong environment
            # and try calling with unsupported arguments, since your
            # stack trace will only show core.py.
            if self.spec:
                reraise(suffix='(for {})'.format(self.spec.id))
            else:
                raise

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Notes:
            EXPERIMENTAL: may be removed in a later version of Gym

            This is a dynamic property in order to avoid refcycles.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        if self._unwrapped is not None:
            return self._unwrapped
        else:
            return self

    def __del__(self):
        self.close()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)

# Space-related abstractions

class Space(object):
    """Defines the observation and action spaces, so you can write generic
    code that applies to any Env. For example, you can choose a random
    action.
    """

    def sample(self, seed=0):
        """
        Uniformly randomly sample a random elemnt of this space
        """
        raise NotImplementedError

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        # By default, assume identity is JSONable
        return sample_n

    def from_jsonable(self, sample_n):
        """Convert a JSONable data type to a batch of samples from this space."""
        # By default, assume identity is JSONable
        return sample_n

class Wrapper(Env):
    # Clear metadata so by default we don't override any keys.
    metadata = {}

    _owns_render = False

    # Make sure self.env is always defined, even if things break
    # early.
    env = None

    def __init__(self, env=None):
        self.env = env
        # Merge with the base metadata
        metadata = self.metadata
        self.metadata = self.env.metadata.copy()
        self.metadata.update(metadata)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self._spec = self.env.spec
        self._unwrapped = self.env.unwrapped

        self._update_wrapper_stack()
        if env and env._configured:
            logger.warning("Attempted to wrap env %s after .configure() was called.", env)

    def _update_wrapper_stack(self):
        """
        Keep a list of all the wrappers that have been appended to the stack.
        """
        self._wrapper_stack = getattr(self.env, '_wrapper_stack', [])
        self._check_for_duplicate_wrappers()
        self._wrapper_stack.append(self)

    def _check_for_duplicate_wrappers(self):
        """Raise an error if there are duplicate wrappers. Can be overwritten by subclasses"""
        if self.class_name() in [wrapper.class_name() for wrapper in self._wrapper_stack]:
            raise error.DoubleWrapperError("Attempted to double wrap with Wrapper: {}".format(self.class_name()))

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _step(self, action):
        return self.env.step(action)

    def _reset(self):
        return self.env.reset()

    def _render(self, mode='human', close=False):
        if self.env is None:
            return
        return self.env.render(mode, close)

    def _close(self):
        if self.env is None:
            return
        return self.env.close()

    def _configure(self, *args, **kwargs):
        return self.env.configure(*args, **kwargs)

    def _seed(self, seed=None):
        return self.env.seed(seed)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def spec(self):
        if self._spec is None:
            self._spec = self.env.spec
        return self._spec

    @spec.setter
    def spec(self, spec):
        # Won't have an env attr while in the __new__ from gym.Env
        if self.env is not None:
            self.env.spec = spec
        self._spec = spec

class ObservationWrapper(Wrapper):
    def _reset(self):
        observation = self.env.reset()
        return self._observation(observation)

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        return self._observation(observation)

    def _observation(self, observation):
        raise NotImplementedError

class RewardWrapper(Wrapper):
    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        return self._reward(reward)

    def _reward(self, reward):
        raise NotImplementedError

class ActionWrapper(Wrapper):
    def _step(self, action):
        action = self.action(action)
        return self.env.step(action)

    def action(self, action):
        return self._action(action)

    def _action(self, action):
        raise NotImplementedError

    def reverse_action(self, action):
        return self._reverse_action(action)

    def _reverse_action(self, action):
        raise NotImplementedError
