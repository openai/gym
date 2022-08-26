from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import make, register, registry, spec

# Hook to load plugins from entry points
_load_env_plugins()


# Classic
# ----------------------------------------

register(
    id="CartPole-v0",
    entry_point="gym.envs.classic_control.cartpole:CartPoleEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id="CartPole-v1",
    entry_point="gym.envs.classic_control.cartpole:CartPoleEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id="MountainCar-v0",
    entry_point="gym.envs.classic_control.mountain_car:MountainCarEnv",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="MountainCarContinuous-v0",
    entry_point="gym.envs.classic_control.continuous_mountain_car:Continuous_MountainCarEnv",
    max_episode_steps=999,
    reward_threshold=90.0,
)

register(
    id="Pendulum-v1",
    entry_point="gym.envs.classic_control.pendulum:PendulumEnv",
    max_episode_steps=200,
)

register(
    id="Acrobot-v1",
    entry_point="gym.envs.classic_control.acrobot:AcrobotEnv",
    reward_threshold=-100.0,
    max_episode_steps=500,
)

# Box2d
# ----------------------------------------

register(
    id="LunarLander-v2",
    entry_point="gym.envs.box2d.lunar_lander:LunarLander",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderContinuous-v2",
    entry_point="gym.envs.box2d.lunar_lander:LunarLander",
    kwargs={"continuous": True},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="BipedalWalker-v3",
    entry_point="gym.envs.box2d.bipedal_walker:BipedalWalker",
    max_episode_steps=1600,
    reward_threshold=300,
)

register(
    id="BipedalWalkerHardcore-v3",
    entry_point="gym.envs.box2d.bipedal_walker:BipedalWalker",
    kwargs={"hardcore": True},
    max_episode_steps=2000,
    reward_threshold=300,
)

register(
    id="CarRacing-v2",
    entry_point="gym.envs.box2d.car_racing:CarRacing",
    max_episode_steps=1000,
    reward_threshold=900,
)

# Toy Text
# ----------------------------------------

register(
    id="Blackjack-v1",
    entry_point="gym.envs.toy_text.blackjack:BlackjackEnv",
    kwargs={"sab": True, "natural": False},
)

register(
    id="FrozenLake-v1",
    entry_point="gym.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "4x4"},
    max_episode_steps=100,
    reward_threshold=0.70,  # optimum = 0.74
)

register(
    id="FrozenLake8x8-v1",
    entry_point="gym.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "8x8"},
    max_episode_steps=200,
    reward_threshold=0.85,  # optimum = 0.91
)

register(
    id="CliffWalking-v0",
    entry_point="gym.envs.toy_text.cliffwalking:CliffWalkingEnv",
)

register(
    id="Taxi-v3",
    entry_point="gym.envs.toy_text.taxi:TaxiEnv",
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
    id="Reacher-v4",
    entry_point="gym.envs.mujoco.reacher_v4:ReacherEnv",
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
    id="Pusher-v4",
    entry_point="gym.envs.mujoco.pusher_v4:PusherEnv",
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
    id="InvertedPendulum-v4",
    entry_point="gym.envs.mujoco.inverted_pendulum_v4:InvertedPendulumEnv",
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
    id="InvertedDoublePendulum-v4",
    entry_point="gym.envs.mujoco.inverted_double_pendulum_v4:InvertedDoublePendulumEnv",
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
    id="HalfCheetah-v4",
    entry_point="gym.envs.mujoco.half_cheetah_v4:HalfCheetahEnv",
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
    id="Hopper-v4",
    entry_point="gym.envs.mujoco.hopper_v4:HopperEnv",
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
    id="Swimmer-v4",
    entry_point="gym.envs.mujoco.swimmer_v4:SwimmerEnv",
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
    id="Walker2d-v4",
    max_episode_steps=1000,
    entry_point="gym.envs.mujoco.walker2d_v4:Walker2dEnv",
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
    id="Ant-v4",
    entry_point="gym.envs.mujoco.ant_v4:AntEnv",
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
    id="Humanoid-v4",
    entry_point="gym.envs.mujoco.humanoid_v4:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidStandup-v2",
    entry_point="gym.envs.mujoco:HumanoidStandupEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidStandup-v4",
    entry_point="gym.envs.mujoco.humanoidstandup_v4:HumanoidStandupEnv",
    max_episode_steps=1000,
)
