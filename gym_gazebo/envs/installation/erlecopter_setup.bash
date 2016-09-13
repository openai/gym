#!/bin/bash

if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:"`pwd`/../assets/models >> ~/.bashrc'
else
  bash -c 'sed "s,GAZEBO_MODEL_PATH=[^;]*,'GAZEBO_MODEL_PATH=`pwd`/../assets/models'," -i ~/.bashrc'
fi

if [ -z "$GYM_GAZEBO_WORLD_CIRCLE" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_CIRCLE="`pwd`/../assets/worlds/circle.world >> ~/.bashrc'
else
  bash -c 'sed "s,GYM_GAZEBO_WORLD_CIRCLE=[^;]*,'GYM_GAZEBO_WORLD_CIRCLE=`pwd`/../assets/worlds/circle.world'," -i ~/.bashrc'
fi

if [ -z "$ARDUPILOT_PATH" ]; then
  bash -c 'echo "export ARDUPILOT_PATH="`pwd`/apm/ardupilot >> ~/.bashrc'
else
  bash -c 'sed "s,ARDUPILOT_PATH=[^;]*,'ARDUPILOT_PATH=`pwd`/apm/ardupilot'," -i ~/.bashrc'
fi

if [ -z "$ERLE_COPTER_PARAM_PATH" ]; then
	bash -c 'echo "export ERLE_COPTER_PARAM_PATH="`pwd`/../assets/params/Erle-Copter.param >> ~/.bashrc'
else
	bash -c 'sed "s,ERLE_COPTER_PARAM_PATH=[^;]*,'ERLE_COPTER_PARAM_PATH=`pwd`/../assets/params/Erle-Copter.param'," -i ~/.bashrc'
fi

#copy altered urdf model
cp -r ../assets/urdf/erlecopter/* catkin_ws/src/ardupilot_sitl_gazebo_plugin/ardupilot_sitl_gazebo_plugin/urdf/

exec bash # reload bash