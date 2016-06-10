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
