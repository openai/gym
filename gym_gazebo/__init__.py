import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Gazebo
# ----------------------------------------

# Turtlebot envs
register(
    id='GazeboMazeTurtlebotLidar-v0',
    entry_point='gym_gazebo.envs.turtlebot:GazeboMazeTurtlebotLidarEnv',
    # More arguments here
)
register(
    id='GazeboCircuitTurtlebotLidar-v0',
    entry_point='gym_gazebo.envs.turtlebot:GazeboCircuitTurtlebotLidarEnv',
    # More arguments here
)
register(
    id='GazeboCircuit2TurtlebotLidar-v0',
    entry_point='gym_gazebo.envs.turtlebot:GazeboCircuit2TurtlebotLidarEnv',
    # More arguments here
)
register(
    id='GazeboCircuit2TurtlebotLidarNn-v0',
    entry_point='gym_gazebo.envs.turtlebot:GazeboCircuit2TurtlebotLidarNnEnv',
    # More arguments here
)
register(
    id='GazeboCircuit2cTurtlebotCameraNnEnv-v0',
    entry_point='gym_gazebo.envs.turtlebot:GazeboCircuit2cTurtlebotCameraNnEnv',
    # More arguments here
)
register(
    id='GazeboRoundTurtlebotLidar-v0',
    entry_point='gym_gazebo.envs.turtlebot:GazeboRoundTurtlebotLidarEnv',
    # More arguments here
)

# Erle-Copter envs
register(
    id='GazeboErleCopterHover-v0',
    entry_point='gym_gazebo.envs.erlecopter:GazeboErleCopterHoverEnv',
)

#Erle-Rover envs
register(
    id='GazeboMazeErleRoverLidar-v0',
    entry_point='gym_gazebo.envs.erlerover:GazeboMazeErleRoverLidarEnv',
)

# Modular SCARA
register(
    id='GazeboModularScara3DOF-v0',
    entry_point='gym_gazebo.envs.modular_scara:GazeboModularScara3DOFEnv',
)

register(
    id='GazeboModularScara3DOF-v1',
    entry_point='gym_gazebo.envs.modular_scara:GazeboModularScara3DOFv1Env',
)

register(
    id='GazeboModularScara3DOF-v2',
    entry_point='gym_gazebo.envs.modular_scara:GazeboModularScara3DOFv2Env',
)
register(
    id='GazeboModularScara3DOF-v3',
    entry_point='gym_gazebo.envs.modular_scara:GazeboModularScara3DOFv3Env',
)
