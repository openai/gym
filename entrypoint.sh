#!/bin/bash
set -e

# setup ros environment
# source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/root/ros_catkin_ws/install_isolated/setup.bash"

# set environment variables
export ROS_PORT_SIM="11311"
# export GYM_GAZEBO_ENV_SCARA3="/Users/victor/gym-gazebo/gym_gazebo/envs/assets/worlds/scara_basic.world"
export GYM_GAZEBO_WORLD_CIRCUIT="/root/gym-gazebo/gym_gazebo/envs/assets/worlds/circuit.world"
export GYM_GAZEBO_WORLD_CIRCUIT2="/root/gym-gazebo/gym_gazebo/envs/assets/worlds/circuit2.world"

# # FIXME
# rm /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so

exec "$@"
