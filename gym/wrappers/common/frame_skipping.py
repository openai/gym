import gym

__all__ = ['Skip', 'Skip1', 'Skip2', 'Skip3', 'Skip4', 'Skip5', 'Skip6', 'Skip7', 'Skip8', 'Skip9', 'Skip10']

class Skip(gym.Wrapper):
    """
        Generic common frame skipping wrapper
        Will perform action for `x` additional steps
    """
    def __init__(self, env, repeat_for_x_steps):
        super(Skip, self).__init__(env)
        self.repeat_count = repeat_for_x_steps
        self.record_stepcount = not hasattr(env.unwrapped, 'stepcount')
        if self.record_stepcount:
            env.unwrapped.stepcount = 0

    def _step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < (self.repeat_count + 1) and not done:
            if self.record_stepcount:
                self.env.unwrapped.stepcount += 1
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info

    def _reset(self):
        if self.record_stepcount:
            self.env.unwrapped.stepcount = 0
        return self.env.reset()

    @property
    def stepcount(self):
        return self.env.unwrapped.stepcount


# Shorthand classes, to avoid having to type the repeat_for_x_steps parameter

class Skip1(Skip):
    def __init__(self, env):
        super(Skip1, self).__init__(env, 1)

class Skip2(Skip):
    def __init__(self, env):
        super(Skip2, self).__init__(env, 2)

class Skip3(Skip):
    def __init__(self, env):
        super(Skip3, self).__init__(env, 3)

class Skip4(Skip):
    def __init__(self, env):
        super(Skip4, self).__init__(env, 4)

class Skip5(Skip):
    def __init__(self, env):
        super(Skip5, self).__init__(env, 5)

class Skip6(Skip):
    def __init__(self, env):
        super(Skip6, self).__init__(env, 6)

class Skip7(Skip):
    def __init__(self, env):
        super(Skip7, self).__init__(env, 7)

class Skip8(Skip):
    def __init__(self, env):
        super(Skip8, self).__init__(env, 8)

class Skip9(Skip):
    def __init__(self, env):
        super(Skip9, self).__init__(env, 9)

class Skip10(Skip):
    def __init__(self, env):
        super(Skip10, self).__init__(env, 10)
