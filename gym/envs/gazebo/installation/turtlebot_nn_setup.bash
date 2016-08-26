#!/bin/bash

if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:"`pwd`/../assets/models >> ~/.bashrc'
fi

#Load turtlebot variables. Temporal solution
chmod +x catkin_ws/src/turtlebot_simulator/turtlebot_gazebo/env-hooks/25.turtlebot-gazebo.sh.em
bash catkin_ws/src/turtlebot_simulator/turtlebot_gazebo/env-hooks/25.turtlebot-gazebo.sh.em

#add turtlebot launch environment variable
if [ -z "$GYM_GAZEBO_WORLD_MAZE" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_MAZE="`pwd`/../assets/worlds/maze.world >> ~/.bashrc'
fi
if [ -z "$GYM_GAZEBO_WORLD_CIRCUIT" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_CIRCUIT="`pwd`/../assets/worlds/circuit.world >> ~/.bashrc'
fi
if [ -z "$GYM_GAZEBO_WORLD_CIRCUIT2" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_CIRCUIT2="`pwd`/../assets/worlds/circuit2.world >> ~/.bashrc'
fi
if [ -z "$GYM_GAZEBO_WORLD_CIRCUIT2C" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_CIRCUIT2C="`pwd`/../assets/worlds/circuit2c.world >> ~/.bashrc'
fi
if [ -z "$GYM_GAZEBO_WORLD_ROUND" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_ROUND="`pwd`/../assets/worlds/round.world >> ~/.bashrc'
fi

#copy altered urdf model
cp -r ../assets/urdf/kobuki_nn_urdf/urdf/ catkin_ws/src/kobuki/kobuki_description

#copy laser mesh file
cp ../assets/meshes/lidar_lite_v2_withRay.dae catkin_ws/src/kobuki/kobuki_description/meshes
 
exec bash # reload bash

