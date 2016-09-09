#!/bin/bash

if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:"`pwd`/../assets/models >> ~/.bashrc'
fi

if [ -z "$GYM_GAZEBO_WORLD_CIRCLE" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_CIRCLE="`pwd`/../assets/worlds/circle.world >> ~/.bashrc'
fi

if [ -z "$ARDUPILOT_PATH" ]; then
  bash -c 'echo "export ARDUPILOT_PATH="`pwd`/apm/ardupilot >> ~/.bashrc'
fi

#copy altered urdf model
cp -r ../assets/urdf/erlecopter/* catkin_ws/src/ardupilot_sitl_gazebo_plugin/ardupilot_sitl_gazebo_plugin/urdf/

exec bash # reload bash