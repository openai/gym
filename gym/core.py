import logging
logger = logging.getLogger(__name__)

import numpy as np

from gym import error, monitoring
from gym.utils import closer

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

    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """

    def __new__(cls, *args, **kwargs):
        # We use __new__ since we want the env author to be able to
        # override __init__ without remebering to call super.
        env = super(Env, cls).__new__(cls)
        env._env_closer_id = env_closer.register(env)
        env._closed = False
        env._action_warned = False
        env._configured = False

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

    @property
    def monitor(self):
        """Lazily creates a monitor instance.

        We do this lazily rather than at environment creation time
        since when the monitor closes, we need remove the existing
        monitor but also make it easy to start a new one. We could
        still just forcibly create a new monitor instance on old
        monitor close, but that seems less clean.
        """
        if not hasattr(self, '_monitor'):
            self._monitor = monitoring.Monitor(self)
        return self._monitor

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
        check_bounds = self.metadata.get('step.check_bounds', True)
        if check_bounds and not self._action_warned and not self.action_space.contains(action):
            self._action_warned = True
            logger.warn("Action '{}' is not contained within action space '{}'.".format(action, self.action_space))

        self.monitor._before_step(action)
        observation, reward, done, info = self._step(action)

        done = self.monitor._after_step(observation, reward, done, info)
        return observation, reward, done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        if self.metadata.get('configure.required') and not self._configured:
            raise error.Error("{} requires calling 'configure()' before 'reset()'".format(self))

        self.monitor._before_reset()
        observation = self._reset()
        self.monitor._after_reset(observation)
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

        # Whether to warn if action or observation exceed limits. Turning this
        # off can give a significant performance boost (20-50%) when observation
        # spaces or action spaces are large (e.g. images)
        self.metadata['step.check_bounds'] = kwargs.pop('check_bounds', True)
        return self._configure(*args, **kwargs)

    def __del__(self):
        self.close()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)

# Space-related abstractions

class Space(object):
    """
    Provides a classification state spaces and action spaces,
    so you can write generic code that applies to any Environment.
    E.g. to choose a random action.
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
