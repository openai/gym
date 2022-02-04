from gym.envs.registration import (
    registry,
    register,
    spec,
    load_env_plugins as _load_env_plugins,
)
from typing import overload, TYPE_CHECKING, Any
import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:

    class Literal:
        def __getitem__(self, parameters):
            return Any


if TYPE_CHECKING:
    from gym.envs.classic_control import (
        CartPoleEnv,
        PendulumEnv,
        AcrobotEnv,
        MountainCarEnv,
        Continuous_MountainCarEnv,
    )
    from gym.envs.box2d import (
        LunarLander,
        CarRacing,
        LunarLanderContinuous,
        BipedalWalker,
    )
    from gym.envs.toy_text import TaxiEnv, BlackjackEnv, FrozenLakeEnv, CliffWalkingEnv
    from gym.envs.mujoco import (
        AntEnv,
        HopperEnv,
        StrikerEnv,
        SwimmerEnv,
        Walker2dEnv,
        ThrowerEnv,
        HalfCheetahEnv,
        ReacherEnv,
        PusherEnv,
        InvertedPendulumEnv,
        InvertedDoublePendulumEnv,
    )
    from gym import Env

# Hook to load plugins from entry points
_load_env_plugins()


# Classic
# ----------------------------------------

register(
    id="CartPole-v0",
    entry_point="gym.envs.classic_control:CartPoleEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id="CartPole-v1",
    entry_point="gym.envs.classic_control:CartPoleEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id="MountainCar-v0",
    entry_point="gym.envs.classic_control:MountainCarEnv",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="MountainCarContinuous-v0",
    entry_point="gym.envs.classic_control:Continuous_MountainCarEnv",
    max_episode_steps=999,
    reward_threshold=90.0,
)

register(
    id="Pendulum-v1",
    entry_point="gym.envs.classic_control:PendulumEnv",
    max_episode_steps=200,
)

register(
    id="Acrobot-v1",
    entry_point="gym.envs.classic_control:AcrobotEnv",
    reward_threshold=-100.0,
    max_episode_steps=500,
)

# Box2d
# ----------------------------------------

register(
    id="LunarLander-v2",
    entry_point="gym.envs.box2d:LunarLander",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderContinuous-v2",
    entry_point="gym.envs.box2d:LunarLander",
    kwargs={"continuous": True},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="BipedalWalker-v3",
    entry_point="gym.envs.box2d:BipedalWalker",
    max_episode_steps=1600,
    reward_threshold=300,
)

register(
    id="BipedalWalkerHardcore-v3",
    entry_point="gym.envs.box2d:BipedalWalker",
    kwargs={"hardcore": True},
    max_episode_steps=2000,
    reward_threshold=300,
)

register(
    id="CarRacing-v0",
    entry_point="gym.envs.box2d:CarRacing",
    max_episode_steps=1000,
    reward_threshold=900,
)

# Toy Text
# ----------------------------------------

register(
    id="Blackjack-v1",
    entry_point="gym.envs.toy_text:BlackjackEnv",
    kwargs={"sab": True, "natural": False},
)

register(
    id="FrozenLake-v1",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name": "4x4"},
    max_episode_steps=100,
    reward_threshold=0.70,  # optimum = 0.74
)

register(
    id="FrozenLake8x8-v1",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name": "8x8"},
    max_episode_steps=200,
    reward_threshold=0.85,  # optimum = 0.91
)

register(
    id="CliffWalking-v0",
    entry_point="gym.envs.toy_text:CliffWalkingEnv",
)

register(
    id="Taxi-v3",
    entry_point="gym.envs.toy_text:TaxiEnv",
    reward_threshold=8,  # optimum = 8.46
    max_episode_steps=200,
)

# Mujoco
# ----------------------------------------

# 2D

register(
    id="Reacher-v2",
    entry_point="gym.envs.mujoco:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id="Pusher-v2",
    entry_point="gym.envs.mujoco:PusherEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id="Thrower-v2",
    entry_point="gym.envs.mujoco:ThrowerEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id="Striker-v2",
    entry_point="gym.envs.mujoco:StrikerEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id="InvertedPendulum-v2",
    entry_point="gym.envs.mujoco:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id="InvertedDoublePendulum-v2",
    entry_point="gym.envs.mujoco:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id="HalfCheetah-v2",
    entry_point="gym.envs.mujoco:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="HalfCheetah-v3",
    entry_point="gym.envs.mujoco.half_cheetah_v3:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="Hopper-v2",
    entry_point="gym.envs.mujoco:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Hopper-v3",
    entry_point="gym.envs.mujoco.hopper_v3:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Swimmer-v2",
    entry_point="gym.envs.mujoco:SwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id="Swimmer-v3",
    entry_point="gym.envs.mujoco.swimmer_v3:SwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id="Walker2d-v2",
    max_episode_steps=1000,
    entry_point="gym.envs.mujoco:Walker2dEnv",
)

register(
    id="Walker2d-v3",
    max_episode_steps=1000,
    entry_point="gym.envs.mujoco.walker2d_v3:Walker2dEnv",
)

register(
    id="Ant-v2",
    entry_point="gym.envs.mujoco:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Ant-v3",
    entry_point="gym.envs.mujoco.ant_v3:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Humanoid-v2",
    entry_point="gym.envs.mujoco:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="Humanoid-v3",
    entry_point="gym.envs.mujoco.humanoid_v3:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidStandup-v2",
    entry_point="gym.envs.mujoco:HumanoidStandupEnv",
    max_episode_steps=1000,
)

# fmt: off
# Continuous
# ----------------------------------------

@overload
def make(id: Literal["CartPole-v0", "CartPole-v1"], **kwargs) -> "CartPoleEnv": ...
@overload
def make(id: Literal["MountainCar-v0"], **kwargs) -> "MountainCarEnv": ...
@overload
def make(id: Literal["MountainCarContinuous-v0"], **kwargs) -> "Continuous_MountainCarEnv": ...
@overload
def make(id: Literal["Pendulum-v1"], **kwargs) -> "PendulumEnv": ...
@overload
def make(id: Literal["Acrobot-v1"], **kwargs) -> "AcrobotEnv": ...

# Box2d
# ----------------------------------------

@overload
def make(id: Literal["LunarLander-v2", "LunarLanderContinuous-v2"], **kwargs) -> "LunarLander": ...
@overload
def make(id: Literal["BipedalWalker-v3", "BipedalWalkerHardcore-v3"], **kwargs) -> "BipedalWalker": ...
@overload
def make(id: Literal["CarRacing-v0"], **kwargs) -> "CarRacing": ...

# Toy Text
# ----------------------------------------

@overload
def make(id: Literal["Blackjack-v1"], **kwargs) -> "BlackjackEnv": ...
@overload
def make(id: Literal["FrozenLake-v1", "FrozenLake8x8-v1"], **kwargs) -> "FrozenLakeEnv": ...
@overload
def make(id: Literal["CliffWalking-v0"], **kwargs) -> "CliffWalkingEnv": ...
@overload
def make(id: Literal["Taxi-v3"], **kwargs) -> "TaxiEnv": ...

# Mujoco
# ----------------------------------------
@overload
def make(id: Literal["Reacher-v2"], **kwargs) -> "ReacherEnv": ...
@overload
def make(id: Literal["Pusher-v2"], **kwargs) -> "PusherEnv": ...
@overload
def make(id: Literal["Thrower-v2"], **kwargs) -> "ThrowerEnv": ...
@overload
def make(id: Literal["Striker-v2"], **kwargs) -> "StrikerEnv": ...
@overload
def make(id: Literal["InvertedPendulum-v2"], **kwargs) -> "InvertedPendulumEnv": ...
@overload
def make(id: Literal["InvertedDoublePendulum-v2"], **kwargs) -> "InvertedDoublePendulumEnv": ...
@overload
def make(id: Literal["HalfCheetah-v2", "HalfCheetah-v3"], **kwargs) -> "HalfCheetahEnv": ...
@overload
def make(id: Literal["Hopper-v2", "Hopper-v3"], **kwargs) -> "HopperEnv": ...
@overload
def make(id: Literal["Swimmer-v2", "Swimmer-v3"], **kwargs) -> "SwimmerEnv": ...
@overload
def make(id: Literal["Walker2d-v2", "Walker2d-v3"], **kwargs) -> "Walker2dEnv": ...
@overload
def make(id: Literal["Ant-v2"], **kwargs) -> "AntEnv": ...

# ----------------------------------------

@overload
def make(id: str, **kwargs) -> "Env": ...
# fmt: on
def make(id: str, **kwargs) -> "Env":
    return registry.make(id, **kwargs)
