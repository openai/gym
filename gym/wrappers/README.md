# Wrappers (experimental)

This is a placeholder for now: we will likely soon start adding
standardized wrappers for environments. (Only stable and
general-purpose wrappers will be accepted into gym core.)

Note that we may later restructure any of the files, but will keep the
wrappers available at the relevant package's top-level. So for
example, you should access `MyWrapper` as follows:

```
# Will be supported in future releases
from gym.wrappers import my_collection
my_collection.MyWrapper
```

## How to add new wrappers to Gym

1. Write your wrapper in an existing collection or a new collection. All collections are subfolders of `/gym/wrappers'. The wrapper collection should be the same as the environment collection.
2. Import your wrapper into the `__init__.py` file of the collection. This file will be located at `/gym/wrappers/my_collection/__init__.py`. Add `from gym.wrappers.my_collection.my_awesome_wrapper import MyWrapper` to this file.
3. Write a good description of the utility of your wrapper using python docstring format (""" """ under the class definition)


## Quick Tips

- You can access the inner environment with `self._unwrapped`
- You can access the previous layer using `self.env`
- The variables `metadata`, `action_space`, `observation_space`, `reward_range`, and `spec` have been copied to `self` from the previous layer
- Create a wrapped function for at least one of the following: `__init__(self, env)`, `_step`, `_reset`, `_render`, `_close`, `_configure`, or `_seed`
- Your layered function should take its input from the previous layer (`self.env`) and/or the inner layer (`self._unwrapped`)
