# Deprecated environments. We want to keep around the old environment
# specifications, though they may no longer be possible to
# instantiate.

from gym.envs.registration import register

# MuJoCo

register(
    id='Reacher-v0',
    timestep_limit=50
)

register(
    id='InvertedPendulum-v0',
)

register(
    id='InvertedDoublePendulum-v0',
)

register(
    id='HalfCheetah-v0',
)

register(
    id='Hopper-v0',
)

register(
    id='Swimmer-v0',
)

register(
    id='Walker2d-v0',
)

register(
    id='Ant-v0',
)

register(
    id='Humanoid-v0',
)

register(
    id='LunarLander-v0',
)

register(
    id='BipedalWalker-v0',
)

register(
    id='BipedalWalkerHardcore-v0',
)
