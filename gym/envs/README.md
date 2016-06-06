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

## How to add new environments to Gym

1. Write your environment in an existing collection or a new collection. All collections are subfolders of `/gym/envs'.
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
