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