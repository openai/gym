"""Miscellaneous utilities."""
import contextlib
import os

__all__ = ["CloudpickleWrapper", "clear_mpi_env_vars"]


class CloudpickleWrapper:
    """Wrapper that uses cloudpickle to pickle and unpickle the result."""

    def __init__(self, fn: callable):
        """Cloudpickle wrapper for a function."""
        self.fn = fn

    def __getstate__(self):
        """Get the state using `cloudpickle.dumps(self.fn)`."""
        import cloudpickle

        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob):
        """Sets the state with obs."""
        import pickle

        self.fn = pickle.loads(ob)

    def __call__(self):
        """Calls the function `self.fn` with no arguments."""
        return self.fn()


@contextlib.contextmanager
def clear_mpi_env_vars():
    """Clears the MPI of environment variables.

    `from mpi4py import MPI` will call `MPI_Init` by default.
    If the child process has MPI environment variables, MPI will think that the child process
    is an MPI process just like the parent and do bad things such as hang.

    This context manager is a hacky way to clear those environment variables
    temporarily such as when we are starting multiprocessing Processes.

    Yields:
        Yields for the context manager
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ["OMPI_", "PMI_"]:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)
