import numpy as np
import multiprocessing as mp
import time

from gym import logger
from gym.vector.vector_env import VectorEnv
from gym.error import (AlreadySteppingError, AlreadyResettingError,
                       NotSteppingError, NotResettingError, ClosedEnvironmentError)
from gym.vector.utils import (create_shared_memory, create_empty_array,
                              write_to_shared_memory, read_from_shared_memory,
                              concatenate, CloudpickleWrapper, clear_mpi_env_vars)


class AsyncVectorEnv(VectorEnv):
    """Vectorized environment that runs multiple environments in parallel. It
    uses `multiprocessing` processes, and pipes for communication.

    Parameters
    ----------
    env_fns : iterable of callable
        Functions that create the environments.

    observation_space : `gym.spaces.Space` instance, optional
        Observation space of a single environment. If `None`, then the
        observation space of the first environment is taken.

    action_space : `gym.spaces.Space` instance, optional
        Action space of a single environment. If `None`, then the action space
        of the first environment is taken.

    shared_memory : bool (default: `True`)
        If `True`, then the observations from the worker processes are
        communicated back through shared variables. This can improve the
        efficiency if the observations are large (e.g. images).

    copy : bool (default: `True`)
        If `True`, then the `reset` and `step` methods return a copy of the
        observations.

    context : str, optional
        Context for multiprocessing. If `None`, then the default context is used.
    """
    def __init__(self, env_fns, observation_space=None, action_space=None,
                 shared_memory=True, copy=True, context=None):
        ctx = mp.get_context(context)
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy

        if (observation_space is None) or (action_space is None):
            dummy_env = env_fns[0]()
            observation_space = observation_space or dummy_env.observation_space
            action_space = action_space or dummy_env.action_space
            dummy_env.close()
            del dummy_env
        super(AsyncVectorEnv, self).__init__(num_envs=len(env_fns),
            observation_space=observation_space, action_space=action_space)

        if self.shared_memory:
            _obs_buffer = create_shared_memory(self.single_observation_space,
                n=self.num_envs)
            self.observations = read_from_shared_memory(_obs_buffer,
                self.single_observation_space, n=self.num_envs)
        else:
            _obs_buffer = None
            self.observations = create_empty_array(self.single_observation_space,
                n=self.num_envs, fn=np.empty)

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = _worker_shared_memory if self.shared_memory else _worker
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(target=target,
                    name='Worker<{0}>-{1}'.format(type(self).__name__, idx),
                    args=(idx, CloudpickleWrapper(env_fn), child_pipe,
                    parent_pipe, _obs_buffer, self.error_queue))

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.deamon = True
                process.start()
                child_pipe.close()

        self.waiting_reset = False
        self.waiting_step = False

    def seed(self, seeds=None):
        self._assert_is_running()
        if seeds is None:
            seeds = [None for _ in range(self.num_envs)]
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        assert len(seeds) == self.num_envs

        for pipe, seed in zip(self.parent_pipes, seeds):
            pipe.send(('seed', seed))
        for pipe in self.parent_pipes:
            pipe.recv()

    def reset_async(self):
        self._assert_is_running()
        if self.waiting_step:
            logger.warn('Calling `reset_async` while waiting for a pending '
                'call to `step` to complete. Cancelling the previous call to '
                '`step`...')
            try:
                self.step_wait(timeout=0)
            except mp.TimeoutError:
                pass

        if self.waiting_reset:
            logger.error('Calling `reset_async` while waiting for a pending '
                'call to `reset` to complete. Closing `{0}`...'.format(self))
            self.close(terminate=True)
            raise AlreadyResettingError('Calling `reset_async` while waiting '
                'for a pending call to `reset` to complete.')

        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        self.waiting_reset = True

    def reset_wait(self, timeout=None):
        """
        Parameters
        ----------
        timeout : int, optional
            Number of seconds before the call to `reset_wait` times out. If
            `None`, the call to `reset_wait` never times out.

        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.
        """
        self._assert_is_running()
        if self.waiting_step or (not self.waiting_reset):
            logger.error('Calling `reset_wait` without any prior call to '
                '`reset_async`. Closing `{0}`...'.format(type(self).__name__))
            self.close(terminate=True)
            raise NotResettingError('Calling `reset_wait` without any prior '
                'call to `reset_async`.')

        if not self.poll(timeout):
            self.waiting_reset = False
            raise mp.TimeoutError('The call to `reset_wait` has timed out after '
                '{0} second{1}.'.format(timeout, 's' if timeout > 1 else ''))

        observations_list = [pipe.recv() for pipe in self.parent_pipes]
        self.waiting_reset = False
        self._raise_if_errors()

        if not self.shared_memory:
            concatenate(observations_list, self.observations,
                self.single_observation_space)

        return np.copy(self.observations) if self.copy else self.observations

    def step_async(self, actions):
        """
        Parameters
        ----------
        actions : iterable of samples from `action_space`
            List of actions.
        """
        self._assert_is_running()
        if self.waiting_reset:
            logger.warn('Calling `step_async` while waiting for a pending '
                'call to `reset` to complete. Waiting for the previous call '
                'to `reset`...')
            try:
                self.reset_wait()
            except mp.TimeoutError:
                pass

        if self.waiting_step:
            logger.error('Calling `step_async` while waiting for a pending '
                'call to `step` to complete. Closing `{0}`...'.format(
                type(self).__name__))
            self.close(terminate=True)
            raise AlreadySteppingError('Calling `step_async` while waiting for '
                'a pending call to `step` to complete.')

        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(('step', action))
        self.waiting_step = True

    def step_wait(self, timeout=None):
        """
        Parameters
        ----------
        timeout : int, optional
            Number of seconds before the call to `step_wait` times out. If
            `None`, the call to `step_wait` never times out.

        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.

        rewards : `np.ndarray` instance (dtype `np.float_`)
            A vector of rewards from the vectorized environment.

        dones : `np.ndarray` instance (dtype `np.bool_`)
            A vector whose entries indicate whether the episode has ended.

        infos : list of dict
            A list of auxiliary diagnostic informations.
        """
        self._assert_is_running()
        if self.waiting_reset or not self.waiting_step:
            logger.error('Calling `step_wait` without any prior call to '
                '`step_async`. Closing `{0}`...'.format(type(self).__name__))
            self.close(terminate=True)
            raise NotSteppingError('Calling `step_wait` without any prior call '
                'to `step_async`.')

        if not self.poll(timeout):
            self.waiting_step = False
            raise mp.TimeoutError('The call to `step_wait` has timed out after '
                '{0} second{1}.'.format(timeout, 's' if timeout > 1 else ''))

        results = [pipe.recv() for pipe in self.parent_pipes]
        self.waiting_step = False
        self._raise_if_errors()
        observations_list, rewards, dones, infos = zip(*results)

        if not self.shared_memory:
            concatenate(observations_list, self.observations,
                self.single_observation_space)

        return (np.copy(self.observations) if self.copy else self.observations,
                np.array(rewards), np.array(dones, dtype=np.bool_), infos)

    def poll(self, timeout=None):
        self._assert_is_running()
        if timeout is not None:
            end_time = time.time() + timeout
        delta = None
        for pipe in self.parent_pipes:
            if timeout is not None:
                delta = max(end_time - time.time(), 0)
            if not pipe.poll(delta):
                break
        else:
            return True
        return False

    def close_extras(self, timeout=None, terminate=False):
        timeout = 0 if terminate else timeout
        try:
            if self.waiting_reset:
                logger.warn('Calling `close` while waiting for a pending '
                    'call to `reset` to complete.')
                self.reset_wait(timeout)

            if self.waiting_step:
                logger.warn('Calling `close` while waiting for a pending '
                    'call to `step` to complete.')
                self.step_wait(timeout)
        except mp.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes:
                pipe.send(('close', None))
            for pipe in self.parent_pipes:
                pipe.recv()

        for pipe in self.parent_pipes:
            pipe.close()
        for process in self.processes:
            process.join()

    def _assert_is_running(self):
        if self.closed:
            raise ClosedEnvironmentError('Trying to operate on `{0}`, after a '
                'call to `close()`.'.format(type(self).__name__))

    def _raise_if_errors(self):
        if not self.error_queue.empty():
            while not self.error_queue.empty():
                index, exctype, value = self.error_queue.get()
                logger.error('Received the following error from Worker-{0}: '
                    '{1}: {2}'.format(index, exctype.__name__, value))
                logger.error('Shutting down Worker-{0}.'.format(index))
                self.parent_pipes[index].close()
                self.parent_pipes[index] = None
            logger.error('Raising the last exception back to the main process.')
            raise exctype(value)

    def __del__(self):
        self.close(terminate=True)


def _worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation = env.reset()
                pipe.send(observation)
            elif command == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                pipe.send((observation, reward, done, info))
            elif command == 'seed':
                env.seed(data)
                pipe.send(None)
            elif command == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError('Received unknown command `{0}`. Must '
                    'be one of {`reset`, `step`, `seed`, `close`}.'.format(command))
    except Exception:
        import sys
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send(None)
    finally:
        env.close()


def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory,
                                       observation_space)
                pipe.send(None)
            elif command == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory,
                                       observation_space)
                pipe.send((None, reward, done, info))
            elif command == 'seed':
                env.seed(data)
                pipe.send(None)
            elif command == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError('Received unknown command `{0}`. Must '
                    'be one of {`reset`, `step`, `seed`, `close`}.'.format(command))
    except Exception:
        import sys
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send(None)
    finally:
        env.close()
