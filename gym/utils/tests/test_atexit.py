from gym.utils.atexit_utils import CloseRegistry

class CloseObject(object):
    close_called = False
    def close(self):
        self.close_called = True

def test_register_unregister():
    registry = CloseRegistry()
    c1 = CloseObject()
    c2 = CloseObject()

    assert not c1.close_called
    assert not c2.close_called
    registry.register(c1)
    id2 = registry.register(c2)

    registry.unregister(id2)
    registry.close()
    assert c1.close_called
    assert not c2.close_called
