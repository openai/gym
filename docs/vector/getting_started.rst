.. automodule:: gym.vector
    :noindex:

Getting Started
===============

Creating a vectorized environment
---------------------------------

To create a vectorized environment that runs multiple sub-environments, you can wrap your sub-environments inside :class:`gym.vector.SyncVectorEnv` (for sequential execution), or :class:`gym.vector.AsyncVectorEnv` (for parallel execution, with `multiprocessing`_). These vectorized environments take as input a list of callable specifying how the sub-environments are created.

.. code-block::

    >>> envs = gym.vector.AsyncVectorEnv([
    ...     lambda: gym.make("CartPole-v1"),
    ...     lambda: gym.make("CartPole-v1"),
    ...     lambda: gym.make("CartPole-v1")
    ... ])

Alternatively, to create a vectorized environment of multiple copies of the same registered sub-environment, you can use the function :func:`gym.vector.make`.

.. code-block::

    >>> envs = gym.vector.make("CartPole-v1", num_envs=3)  # Equivalent

.. note::
    To enable automatic batching of actions and observations, all the sub-environments must share the same :obj:`action_space` and :obj:`observation_space`. However, all the sub-environments are not required to be exact copies of one another. For example, you can run 2 instances of ``Pendulum-v0`` with different values of the gravity in a vectorized environment with

    .. code-block::

        >>> env = gym.vector.AsyncVectorEnv([
        ...     lambda: gym.make("Pendulum-v0", g=9.81),
        ...     lambda: gym.make("Pendulum-v0", g=1.62)
        ... ])

    See also :ref:`Observation & Action spaces` for more information about automatic batching.

.. warning::
    When using :class:`AsyncVectorEnv` with either the ``spawn`` or ``forkserver`` start methods, you must wrap your code containing the vectorized environment with ``if __name__ == "__main__":``. See `this documentation <https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods>`_ for more information.

    .. code-block::

        if __name__ == "__main__":
            envs = gym.vector.make("CartPole-v1", num_envs=3, context="spawn")

Working with vectorized environments
------------------------------------

While standard Gym environments take a single action and return a single observation (with a reward, and boolean indicating termination), vectorized environments take a *batch of actions* as input, and return a *batch of observations*, together with an array of rewards and booleans indicating if the episode ended in each sub-environment.

.. code-block::

    >>> envs = gym.vector.make("CartPole-v1", num_envs=3)
    >>> envs.reset()
    array([[ 0.00198895, -0.00569421, -0.03170966,  0.00126465],
           [-0.02658334,  0.00755256,  0.04376719, -0.00266695],
           [-0.02898625,  0.04779156,  0.02686412, -0.01298284]],
          dtype=float32)

    >>> actions = np.array([1, 0, 1])
    >>> observations, rewards, dones, infos = envs.step(actions)

    >>> observations
    array([[ 0.00187507,  0.18986781, -0.03168437, -0.301252  ],
           [-0.02643229, -0.18816885,  0.04371385,  0.3034975 ],
           [-0.02803041,  0.24251814,  0.02660446, -0.29707024]],
          dtype=float32)
    >>> rewards
    array([1., 1., 1.])
    >>> dones
    array([False, False, False])
    >>> infos
    ({}, {}, {})

Vectorized environments are compatible with any sub-environment, regardless of the action and observation spaces (e.g. container spaces like :class:`~gym.spaces.Dict`, or any arbitrarily nested spaces). In particular, vectorized environments can automatically batch the observations returned by :meth:`~VectorEnv.reset` and :meth:`~VectorEnv.step` for any standard Gym space (e.g. :class:`~gym.spaces.Box`, :class:`~gym.spaces.Discrete`, :class:`~gym.spaces.Dict`, or any nested structure thereof). Similarly, vectorized environments can take batches of actions from any standard Gym space.

.. code-block::

    >>> class DictEnv(gym.Env):
    ...     observation_space = gym.spaces.Dict({
    ...         "position": gym.spaces.Box(-1., 1., (3,), np.float32),
    ...         "velocity": gym.spaces.Box(-1., 1., (2,), np.float32)
    ...     })
    ...     action_space = gym.spaces.Dict({
    ...         "fire": gym.spaces.Discrete(2),
    ...         "jump": gym.spaces.Discrete(2),
    ...         "acceleration": gym.spaces.Box(-1., 1., (2,), np.float32)
    ...     })
    ...
    ...     def reset(self):
    ...         return self.observation_space.sample()
    ...
    ...     def step(self, action):
    ...         observation = self.observation_space.sample()
    ...         return (observation, 0., False, {})

    >>> envs = gym.vector.AsyncVectorEnv([lambda: DictEnv()] * 3)
    >>> envs.observation_space
    Dict(position:Box(-1.0, 1.0, (3, 3), float32), velocity:Box(-1.0, 1.0, (3, 2), float32))
    >>> envs.action_space
    Dict(fire:MultiDiscrete([2 2 2]), jump:MultiDiscrete([2 2 2]), acceleration:Box(-1.0, 1.0, (3, 2), float32))

    >>> envs.reset()
    >>> actions = {
    ...     "fire": np.array([1, 1, 0]),
    ...     "jump": np.array([0, 1, 0]),
    ...     "acceleration": np.random.uniform(-1., 1., size=(3, 2))
    ... }
    >>> observations, rewards, dones, infos = envs.step(actions)
    >>> observations
    {"position": array([[-0.5337036 ,  0.7439302 ,  0.41748118],
                        [ 0.9373266 , -0.5780453 ,  0.8987405 ],
                        [-0.917269  , -0.5888639 ,  0.812942  ]], dtype=float32),
    "velocity": array([[ 0.23626241, -0.0616814 ],
                       [-0.4057572 , -0.4875375 ],
                       [ 0.26341468,  0.72282314]], dtype=float32)}

.. note::
    The sub-environments inside a vectorized environment automatically call :obj:`reset` at the end of an episode. In the following example, the episode of the 3rd sub-environment ends after 2 steps (the agent fell in a hole), and the sub-environment gets reset (observation ``0``).

    .. code-block::

        >>> envs = gym.vector.make("FrozenLake-v1", num_envs=3, is_slippery=False)
        >>> envs.reset()
        array([0, 0, 0])
        >>> observations, rewards, dones, infos = envs.step(np.array([1, 2, 2]))
        >>> observations, rewards, dones, infos = envs.step(np.array([1, 2, 1]))

        >>> dones
        array([False, False,  True])
        >>> observations
        array([8, 2, 0])

Observation & Action spaces
---------------------------

Like any Gym environment, vectorized environments contain two properties :attr:`~VectorEnv.observation_space` and :attr:`~VectorEnv.action_space` to specify the observation and action spaces of the environment. Since vectorized environments operate on multiple sub-environments, where the observations and actions of sub-environments are batched together, the observation and action spaces are adequately batched as well so that the input actions are valid elements of :attr:`~VectorEnv.action_space`, and the observations are valid elements of :attr:`~VectorEnv.observation_space`.

.. code-block::

    >>> envs = gym.vector.make("CartPole-v1", num_envs=3)
    >>> envs.observation_space
    Box([[-4.8 ...]], [[4.8 ...]], (3, 4), float32)
    >>> envs.action_space
    MultiDiscrete([2 2 2])

.. note::
    In order to appropriately batch the observations and actions in vectorized environments, the observation and action spaces of all the sub-environments are required to be identical.

    .. code-block::

        >>> envs = gym.vector.AsyncVectorEnv([
        ...     lambda: gym.make("CartPole-v1"),
        ...     lambda: gym.make("MountainCar-v0")
        ... ])
        RuntimeError: Some environments have an observation space different from `Box([-4.8 ...], [4.8 ...], (4,), float32)`. In order to batch observations, the observation spaces from all environments must be equal.

However, sometimes it may be handy to have access to the observation and action spaces of a sub-environment, and not the batched spaces. You can access those with the properties :attr:`~VectorEnv.single_observation_space` and :attr:`~VectorEnv.single_action_space` of the vectorized environment.

.. code-block::

    >>> envs = gym.vector.make("CartPole-v1", num_envs=3)
    >>> envs.single_observation_space
    Box([-4.8 ...], [4.8 ...], (4,), float32)
    >>> envs.single_action_space
    Discrete(2)

This is convenient, for example, if you instantiate a policy. In the following example, we used :attr:`~VectorEnv.single_observation_space` and :attr:`~VectorEnv.single_action_space` to define the weights of a linear policy. Note that thanks to the vectorized environment, you can apply the policy directly to the whole batch of observations with a single call to :obj:`policy`.

.. code-block::

    >>> from gym.spaces.utils import flatdim
    >>> from scipy.special import softmax

    >>> def policy(weights, observations):
    ...     logits = np.dot(observations, weights)
    ...     return softmax(logits, axis=1)

    >>> envs = gym.vector.make("CartPole-v1", num_envs=3)
    >>> weights = np.random.randn(
    ...     flatdim(envs.single_observation_space),
    ...     envs.single_action_space.n
    ... )
    >>> observations = envs.reset()
    >>> actions = policy(weights, observations).argmax(axis=1)
    >>> observations, rewards, dones, infos = envs.step(actions)

.. _multiprocessing: https://docs.python.org/3/library/multiprocessing.html