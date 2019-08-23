# Wrappers

Wrappers are used to transform an environment in a modular way:

```python
env = gym.make('Pong-v0')
env = MyWrapper(env)
```

Note that we may later restructure any of the files in this directory,
but will keep the wrappers available at the wrappers' top-level
folder. So for example, you should access `MyWrapper` as follows:

```python
from gym.wrappers import MyWrapper
```

## Quick tips for writing your own wrapper

- Don't forget to call `super(class_name, self).__init__(env)` if you override the wrapper's `__init__` function
- You can access the inner environment with `self.unwrapped`
- You can access the previous layer using `self.env`
- The variables `metadata`, `action_space`, `observation_space`, `reward_range`, and `spec` are copied to `self` from the previous layer
- Create a wrapped function for at least one of the following: `__init__(self, env)`, `step`, `reset`, `render`, `close`, or `seed`
- Your layered function should take its input from the previous layer (`self.env`) and/or the inner layer (`self.unwrapped`)
