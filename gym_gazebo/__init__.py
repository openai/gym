import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Gazebo
# ----------------------------------------

# Turtlebot envs
register(
    id='GazeboMazeTurtlebotLidar-v0',
    entry_point='gym_gazebo.envs:GazeboMazeTurtlebotLidarEnv',
    # More arguments here
)
register(
    id='GazeboCircuitTurtlebotLidar-v0',
    entry_point='gym_gazebo.envs:GazeboCircuitTurtlebotLidarEnv',
    # More arguments here
)
register(
    id='GazeboCircuit2TurtlebotLidar-v0',
    entry_point='gym_gazebo.envs:GazeboCircuit2TurtlebotLidarEnv',
    # More arguments here
)
register(
    id='GazeboCircuit2TurtlebotLidarNn-v0',
    entry_point='gym_gazebo.envs:GazeboCircuit2TurtlebotLidarNnEnv',
    # More arguments here
)
register(
    id='GazeboCircuit2cTurtlebotCameraNnEnv-v0',
    entry_point='gym_gazebo.envs:GazeboCircuit2cTurtlebotCameraNnEnv',
    # More arguments here
)
register(
    id='GazeboRoundTurtlebotLidar-v0',
    entry_point='gym_gazebo.envs:GazeboRoundTurtlebotLidarEnv',
    # More arguments here
)

# Erle-Copter envs
register(
    id='GazeboErleCopterHover-v0',
    entry_point='gym_gazebo.envs:GazeboErleCopterHoverEnv',
)

#Erle-Rover envs
register(
    id='GazeboMazeErleRoverLidar-v0',
    entry_point='gym_gazebo.envs:GazeboMazeErleRoverLidarEnv',
)