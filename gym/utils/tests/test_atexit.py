from gym.utils.closer import Closer

class Closeable(object):
    close_called = False
    def close(self):
        self.close_called = True

def test_register_unregister():
    registry = Closer(atexit_register=False)
    c1 = Closeable()
    c2 = Closeable()

    assert not c1.close_called
    assert not c2.close_called
    registry.register(c1)
    id2 = registry.register(c2)

    registry.unregister(id2)
    registry.close()
    assert c1.close_called
    assert not c2.close_called
