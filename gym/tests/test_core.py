from gym import core

# This looks like a pretty trivial, but given our usage of
# __new__, it's worth having.
def test_env_instantiation():
    class ArgumentEnv(core.Env):
        calls = 0
        def __init__(self, arg):
            self.calls += 1
            self.arg = arg

    env = ArgumentEnv('arg')
    assert env.arg == 'arg'
    assert env.calls == 1

# Monitors are kept around until the termination of the current process,
# unless explicitly closed. Ensure that the monitor does not keep the env
# itself around, too.
def test_monitor_does_not_block_gc():
    class RefcountingEnv(core.Env):
        instances = 0

        def __init__(self):
            RefcountingEnv.instances += 1

        def __del__(self):
            RefcountingEnv.instances -= 1

        def _reset(self):
            pass

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

# Our code should ensure that close is only called once.
def test_double_close():
    class CloseCountingEnv(core.Env):
        def __init__(self):
            self.close_count = 0

        def _close(self):
            self.close_count += 1

    env = CloseCountingEnv()
    assert env.close_count == 0
    env.close()
    assert env.close_count == 1
    env.close()
    assert env.close_count == 1
