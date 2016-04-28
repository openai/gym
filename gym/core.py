import logging
import numpy as np

from gym import error, monitoring

# Env-related abstractions

class Env(object):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.

    The main API methods that users of this class need to know are:

        reset
        step
        render

    When implementing an environment, override the following methods
    in your subclass:

        _step
        _reset
        _render

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations

    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality to over time.

    """

    # Set this in SOME subclasses
    metadata = {'render.modes': []}

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

    # Will be automatically set when creating an environment via
    # 'make'.
    spec = None

    @property
    def monitor(self):
        if not hasattr(self, '_monitor'):
            self._monitor = monitoring.Monitor(self)
        return self._monitor

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Input
        -----
        action : an action provided by the environment

        Outputs
        -------
        (observation, reward, done, info)

        observation (object): agent's observation of the current environment
        reward (float) : amount of reward due to the previous action
        done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
        info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.monitor._before_step(action)
        observation, reward, done, info = self._step(action)
        done = self.monitor._after_step(observation, reward, done, info)
        return observation, reward, done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        Outputs
        -------
        observation (object): the initial observation of the space. (Initial reward is assumed to be 0.)
        """
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
