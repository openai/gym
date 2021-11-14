.. automodule:: gym.vector
    :noindex:

Intermediate Usage
==================

Shared memory
-------------

:class:`AsyncVectorEnv` runs each sub-environment inside an individual process. At each call to :meth:`~AsyncVectorEnv.reset` or :meth:`~AsyncVectorEnv.step`, the observations of all the sub-environments are sent back to the main process. To avoid expensive transfers of data between processes, especially with large observations (e.g. images), :class:`AsyncVectorEnv` uses a shared memory by default (``shared_memory=True``) that processes can write to and read from at minimal cost. This can increase the throughout of the vectorized environment.

.. code-block::

    >>> env_fns = [lambda: gym.make("BreakoutNoFrameskip-v4")] * 5

    >>> envs = gym.vector.AsyncVectorEnv(env_fns, shared_memory=False)
    >>> envs.reset()
    >>> %timeit envs.step(envs.action_space.sample())
    2.23 ms ± 136 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    >>> envs = gym.vector.AsyncVectorEnv(env_fns, shared_memory=True)
    >>> envs.reset()
    >>> %timeit envs.step(envs.action_space.sample())
    1.36 ms ± 15.4 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

Exception handling
------------------

Because sometimes things may not go as planned, the exceptions raised in sub-environments are re-raised in the vectorized environment, even when the sub-environments run in parallel with :class:`AsyncVectorEnv`. This way, you can choose how to handle these exceptions yourself (with ``try ... except``).

.. code-block::

    >>> class ErrorEnv(gym.Env):
    ...     observation_space = gym.spaces.Box(-1., 1., (2,), np.float32)
    ...     action_space = gym.spaces.Discrete(2)
    ...
    ...     def reset(self):
    ...         return np.zeros((2,), dtype=np.float32)
    ...
    ...     def step(self, action):
    ...         if action == 1:
    ...             raise ValueError("An error occurred.")
    ...         observation = self.observation_space.sample()
    ...         return (observation, 0., False, {})

    >>> envs = gym.vector.AsyncVectorEnv([lambda: ErrorEnv()] * 3)
    >>> observations = envs.reset()
    >>> observations, rewards, dones, infos = envs.step(np.array([0, 0, 1]))
    ERROR: Received the following error from Worker-2: ValueError: An error occurred.
    ERROR: Shutting down Worker-2.
    ERROR: Raising the last exception back to the main process.
    ValueError: An error occurred.

.. _multiprocessing: https://docs.python.org/3/library/multiprocessing.html