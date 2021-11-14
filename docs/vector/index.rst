.. automodule:: gym.vector

Vectorized Environments
=======================

.. toctree::
    :hidden:

    getting_started
    intermediate
    advanced
    api_reference

*Vectorized environments* are environments that run multiple (independent) sub-environments, either sequentially, or in parallel using `multiprocessing`_. Vectorized environments take as input a batch of actions, and return a batch of observations. This is particularly useful, for example, when the policy is defined as a neural network that operates over a batch of observations.

Gym provides two types of vectorized environments:

    - :class:`gym.vector.SyncVectorEnv`, where the sub-environment are executed sequentially.
    - :class:`gym.vector.AsyncVectorEnv`, where the sub-environments are executed in parallel using `multiprocessing`_. This creates one process per sub-environment.

.. rubric:: Quickstart

Similar to :func:`gym.make`, you can run a vectorized version of a registered environment using the :func:`gym.vector.make` function. This runs multiple copies of the same environment (in parallel, by default).

The following example runs 3 copies of the ``CartPole-v1`` environment in parallel, taking as input a vector of 3 binary actions (one for each sub-environment), and returning an array of 3 observations stacked along the first dimension, with an array of rewards returned by each sub-environment, and an array of booleans indicating if the episode in each sub-environment has ended.

.. code-block::

    >>> envs = gym.vector.make("CartPole-v1", num_envs=3)
    >>> envs.reset()
    >>> actions = np.array([1, 0, 1])
    >>> observations, rewards, dones, infos = envs.step(actions)

    >>> observations
    array([[ 0.00122802,  0.16228443,  0.02521779, -0.23700266],
           [ 0.00788269, -0.17490888,  0.03393489,  0.31735462],
           [ 0.04918966,  0.19421194,  0.02938497, -0.29495203]],
          dtype=float32)
    >>> rewards
    array([1., 1., 1.])
    >>> dones
    array([False, False, False])
    >>> infos
    ({}, {}, {})

.. note::

    The function :func:`gym.vector.make` is meant to be used only in basic cases (e.g. running multiple copies of the same registered environment). For any other use-cases, please use either the :class:`SyncVectorEnv` for sequential execution, or :class:`AsyncVectorEnv` for parallel execution. These use-cases may include:

        - Running multiple instances of the same environment with different parameters (e.g. ``"Pendulum-v0"`` with different values for the gravity).
        - Running multiple instances of an unregistered environment (e.g. a custom environment)
        - Using a wrapper on some (but not all) sub-environments.

.. _multiprocessing: https://docs.python.org/3/library/multiprocessing.html