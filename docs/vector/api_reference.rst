.. automodule:: gym.vector
    :noindex:

API Reference
=============

.. autofunction:: make

VectorEnv
---------

.. autoclass:: VectorEnv
    
    .. py:property:: action_space

        :type: :class:`gym.spaces.Space`

        The (batched) action space. The input actions of :meth:`step` must be valid elements of :obj:`action_space`.

        .. code-block::

            >>> envs = gym.vector.make("CartPole-v1", num_envs=3)
            >>> envs.action_space
            MultiDiscrete([2 2 2])

    .. py:property:: observation_space

        :type: :class:`gym.spaces.Space`

        The (batched) observation space. The observations returned by :meth:`reset` and :meth:`step` are valid elements of :obj:`observation_space`.

        .. code-block::

            >>> envs = gym.vector.make("CartPole-v1", num_envs=3)
            >>> envs.observation_space
            Box([[-4.8 ...]], [[4.8 ...]], (3, 4), float32)

    .. py:property:: single_action_space

        :type: :class:`gym.spaces.Space`

        The action space of a sub-environment.

        .. code-block::

            >>> envs = gym.vector.make("CartPole-v1", num_envs=3)
            >>> envs.single_action_space
            Discrete(2)

    .. py:property:: single_observation_space

        :type: :class:`gym.spaces.Space`

        The observation space of a sub-environment.

        .. code-block::

            >>> envs = gym.vector.make("CartPole-v1", num_envs=3)
            >>> envs.single_action_space
            Box([-4.8 ...], [4.8 ...], (4,), float32)

    .. automethod:: reset

        .. code-block::

            >>> envs = gym.vector.make("CartPole-v1", num_envs=3)
            >>> envs.reset()
            array([[-0.04456399,  0.04653909,  0.01326909, -0.02099827],
                   [ 0.03073904,  0.00145001, -0.03088818, -0.03131252],
                   [ 0.03468829,  0.01500225,  0.01230312,  0.01825218]],
                  dtype=float32)

    .. automethod:: step

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

    .. automethod:: seed

        .. code-block::

            >>> envs = gym.vector.make("CartPole-v1", num_envs=3)
            >>> envs.seed([1, 3, 5])
            >>> envs.reset()
            array([[ 0.03073904,  0.00145001, -0.03088818, -0.03131252],
                   [ 0.02281231, -0.02475473,  0.02306162,  0.02072129],
                   [-0.03742824, -0.02316945,  0.0148571 ,  0.0296055 ]],
                  dtype=float32)

    .. automethod:: close

AsyncVectorEnv
--------------

.. autoclass:: AsyncVectorEnv

    .. automethod:: reset

        .. note::
            This is equivalent to a call to :meth:`reset_async`, followed by a subsequent call to :meth:`reset_wait` (with no timeout).

    .. automethod:: reset_async
    .. automethod:: reset_wait
    .. automethod:: step

        .. note::
            This is equivalent to a call to :meth:`step_async`, followed by a subsequent call to :meth:`step_wait` (with no timeout).

    .. automethod:: step_async
    .. automethod:: step_wait
    .. automethod:: seed
    .. automethod:: close_extras

SyncVectorEnv
-------------

.. autoclass:: SyncVectorEnv

    .. automethod:: reset
    .. automethod:: step
    .. automethod:: seed
    .. automethod:: close_extras

.. _multiprocessing: https://docs.python.org/3/library/multiprocessing.html