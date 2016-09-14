#!/bin/bash

if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:"`pwd`/../assets/models >> ~/.bashrc'
else
  bash -c 'sed "s,GAZEBO_MODEL_PATH=[^;]*,'GAZEBO_MODEL_PATH=`pwd`/../assets/models'," -i ~/.bashrc'
fi

if [ -z "$ARDUPILOT_PATH" ]; then
  bash -c 'echo "export ARDUPILOT_PATH="`pwd`/apm/ardupilot >> ~/.bashrc'
else
  bash -c 'sed "s,ARDUPILOT_PATH=[^;]*,'ARDUPILOT_PATH=`pwd`/apm/ardupilot'," -i ~/.bashrc'
fi

if [ -z "$ERLE_ROVER_PARAM_PATH" ]; then
	bash -c 'echo "export ERLE_ROVER_PARAM_PATH="`pwd`/../assets/params/Erle-Rover.param >> ~/.bashrc'
else
	bash -c 'sed "s,ERLE_ROVER_PARAM_PATH=[^;]*,'ERLE_ROVER_PARAM_PATH=`pwd`/../assets/params/Erle-Rover.param'," -i ~/.bashrc'
fi

#copy altered urdf model
cp -r ../assets/urdf/erlerover/* catkin_ws/src/ardupilot_sitl_gazebo_plugin/ardupilot_sitl_gazebo_plugin/urdf/

exec bash # reload bash