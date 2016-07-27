from gym import core

class ArgumentEnv(core.Env):
    calls = 0

    def __init__(self, arg):
        self.calls += 1
        self.arg = arg

def test_env_instantiation():
    # This looks like a pretty trivial, but given our usage of
    # __new__, it's worth having.
    env = ArgumentEnv('arg')
    assert env.arg == 'arg'
    assert env.calls == 1

class RefcountingEnv(core.Env):
    instances = 0

    def __init__(self):
        RefcountingEnv.instances += 1

    def __del__(self):
        RefcountingEnv.instances -= 1

    def _reset(self):
        pass

# Monitors are kept around until the termination of the current process,
# unless explicitly closed. Ensure that the monitor does not keep the env
# itself around, too.
def test_monitor_does_not_block_gc():
    def make_and_use_env(call_reset=False):
        env = RefcountingEnv()
        assert 1 == RefcountingEnv.instances
        if call_reset:
            env.reset()

    assert 0 == RefcountingEnv.instances
    make_and_use_env()
    assert 0 == RefcountingEnv.instances

    assert 0 == RefcountingEnv.instances
    make_and_use_env(call_reset=True)
    assert 0 == RefcountingEnv.instances
