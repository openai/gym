import contextlib
import os
import io
import builtins

safe_builtins = {
    'range',
    'complex',
    'set',
    'frozenset',
    'slice',
}


class RestrictedUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        """Only allow safe classes from builtins"""
        if module == "builtins" and name in safe_builtins:
            return getattr(builtins, name)
        """Forbid everything else"""
        raise pickle.UnpicklingError("global '%s.%s' is forbidden" %
                                     (module, name))

def restricted_loads(s):
    """Helper function analogous to pickle.loads()"""
    return RestrictedUnpickler(io.BytesIO(s)).load()

__all__ = ['CloudpickleWrapper', 'clear_mpi_env_vars']

class CloudpickleWrapper(object):
    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob):
        import pickle
        self.fn = pickle.loads(restricted_loads(ob))

    def __call__(self):
        return self.fn()

@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    `from mpi4py import MPI` will call `MPI_Init` by default. If the child
    process has MPI environment variables, MPI will think that the child process
    is an MPI process just like the parent and do bad things such as hang.
    
    This context manager is a hacky way to clear those environment variables
    temporarily such as when we are starting multiprocessing Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)
