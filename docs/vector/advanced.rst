.. automodule:: gym.vector
    :noindex:

Advanced Usage
==============

Custom spaces
-------------

Vectorized environments will batch actions and observations if they are elements from standard Gym spaces, such as :class:`~gym.spaces.Box`, :class:`~gym.spaces.Discrete`, or :class:`~gym.spaces.Dict`. If you create your own environment with a custom action and/or observation space though (inheriting from :class:`gym.Space`), the vectorized environment will not attempt to automatically batch the actions/observations, and instead it will return the raw tuple of elements from all sub-environments.

In the following example, we created a new environment :obj:`SMILESEnv`, whose observations are strings representing the `SMILES`_ notation of a molecular structure, with a custom observation space :obj:`SMILES`. The observations returned by the vectorized environment is a tuple of strings. 

.. code-block::

    >>> class SMILES(gym.Space):
    ...     def __init__(self, symbols):
    ...         super().__init__()
    ...         self.symbols = symbols
    ...
    ...     def __eq__(self, other):
    ...         return self.symbols == other.symbols

    >>> class SMILESEnv(gym.Env):
    ...     observation_space = SMILES("][()CO=")
    ...     action_space = gym.spaces.Discrete(7)
    ...
    ...     def reset(self):
    ...         self._state = "["
    ...         return self._state
    ...
    ...     def step(self, action):
    ...         self._state += self.observation_space.symbols[action]
    ...         reward = done = (action == 0)
    ...         return (self._state, float(reward), done, {})

    >>> envs = gym.vector.AsyncVectorEnv(
    ...     [lambda: SMILESEnv()] * 3,
    ...     shared_memory=False
    ... )
    >>> envs.reset()
    >>> observations, rewards, dones, infos = envs.step(np.array([2, 5, 4]))
    >>> observations
    ('[(', '[O', '[C')

.. warning::
    Custom observation & action spaces may inherit from the :class:`gym.Space` class. However, most use-cases should be covered by the existing space classes (e.g. :class:`~gym.spaces.Box`, :class:`~gym.spaces.Discrete`, etc...), and container classes (:class:`~gym.spaces.Tuple` & :class:`~gym.spaces.Dict`). Moreover, some implementations of Reinforcement Learning algorithms might not handle custom spaces properly. Use custom spaces with care.

.. warning::
    If you use :class:`AsyncVectorEnv` with a custom observation space, you must set ``shared_memory=False``, since shared memory and automatic batching is not compatible with custom spaces. In general if you use custom spaces with :class:`AsyncVectorEnv`, the elements of those spaces must be `pickleable`_.


.. _SMILES: https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system
.. _pickleable: https://docs.python.org/3/library/pickle.html