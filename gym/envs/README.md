# Envs

These are the core integrated environments. Note that we may later
restructure any of the files, but will keep the environments available
at the relevant package's top-level. So for example, you should access
`AntEnv` as follows:

```
# Will be supported in future releases
from gym.envs import mujoco
mujoco.AntEnv
```

Rather than:

```
# May break in future releases
from gym.envs.mujoco import ant
ant.AntEnv
```

## How to create new environments for Gym

* Create a new repo called gym-foo, which should also be a PIP package.

* A good example is https://github.com/openai/gym-soccer.

* It should have at least the following files:
  ```sh
  gym-foo/
    README.md
    setup.py
    gym_foo/
      __init__.py
      envs/
        __init__.py
        foo_env.py
        foo_extrahard_env.py
  ```

* `gym-foo/setup.py` should have:

  ```python
  from setuptools import setup

  setup(name='gym_foo',
        version='0.0.1',
        install_requires=['gym']  # And any other dependencies foo needs
  )
  ```

* `gym-foo/gym_foo/__init__.py` should have:
  ```python
  from gym.envs.registration import register

  register(
      id='foo-v0',
      entry_point='gym_foo.envs:FooEnv',
  )
  register(
      id='foo-extrahard-v0',
      entry_point='gym_foo.envs:FooExtraHardEnv',
  )
  ```

* `gym-foo/gym_foo/envs/__init__.py` should have:
  ```python
  from gym_foo.envs.foo_env import FooEnv
  from gym_foo.envs.foo_extrahard_env import FooExtraHardEnv
  ```

* `gym-foo/gym_foo/envs/foo_env.py` should look something like:
  ```python
  import gym
  from gym import error, spaces, utils
  from gym.utils import seeding

  class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
      ...
    def step(self, action):
      ...
    def reset(self):
      ...
    def render(self, mode='human'):
      ...
    def close(self):
      ...
  ```

* After you have installed your package with `pip install -e gym-foo`, you can create an instance of the environment with `gym.make('gym_foo:foo-v0')`

## How to add new environments to Gym, within this repo (not recommended for new environments)

1. Write your environment in an existing collection or a new collection. All collections are subfolders of `/gym/envs`.
2. Import your environment into the `__init__.py` file of the collection. This file will be located at `/gym/envs/my_collection/__init__.py`. Add `from gym.envs.my_collection.my_awesome_env import MyEnv` to this file.
3. Register your env in `/gym/envs/__init__.py`:

 ```
register(
		id='MyEnv-v0',
		entry_point='gym.envs.my_collection:MyEnv',
)
```

4. Add your environment to the scoreboard in `/gym/scoreboard/__init__.py`:

 ```
add_task(
		id='MyEnv-v0',
		summary="Super cool environment",
		group='my_collection',
		contributor='mygithubhandle',
)
```
