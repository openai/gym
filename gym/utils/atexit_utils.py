import atexit
import threading
import weakref

class CloseRegistry(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.next_id = -1
        self.close_objects = weakref.WeakValueDictionary()

    def generate_next_id(self):
        with self.lock:
            self.next_id += 1
            return self.next_id

    def register(self, close_object):
        # Call close_registry.register() on an object with a close method to
        # register it for close on exit. Save the id, and inside the object's
        # close() method, call unregister with the same id to avoid double
        # closing (this isn't perfectly safe e.g. in a multithreaded world but
        # should be good enough for now)
        next_id = self.generate_next_id()
        self.close_objects[next_id] = close_object
        return next_id

    def unregister(self, id):
        # envs not created with make() are not registered
        if id is not None and id in self.close_objects:
            del self.close_objects[id]

    def close(self):
        # Explicitly fetch all monitors first so that they can't disappear while
        # we iterate. cf. http://stackoverflow.com/a/12429620
        objects = list(self.close_objects.items())
        for _, obj in objects:
            obj.close()

env_close_registry = CloseRegistry()

@atexit.register
def close_all_envs():
    env_close_registry.close()
