#!/bin/bash

if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:"`pwd`/../assets/models >> ~/.bashrc'
else
  bash -c 'sed "s,GAZEBO_MODEL_PATH=[^;]*,'GAZEBO_MODEL_PATH=`pwd`/../assets/models'," -i ~/.bashrc'
fi

# add modular scara environment variables
if [ -z "$GYM_GAZEBO_ENV_SCARA3" ]; then
  bash -c 'echo "export GYM_GAZEBO_ENV_SCARA3="`pwd`/../assets/worlds/scara_basic.world >> ~/.bashrc'
else
  bash -c 'sed "s,GYM_GAZEBO_ENV_SCARA3=[^;]*,'GYM_GAZEBO_ENV_SCARA3=`pwd`/../assets/worlds/scara_basic.world'," -i ~/.bashrc'
fi

if [ -z "$GYM_GAZEBO_ENV_SCARA4" ]; then
  bash -c 'echo "export GYM_GAZEBO_ENV_SCARA4="`pwd`/../assets/worlds/scara_basic.world >> ~/.bashrc'
else
  bash -c 'sed "s,GYM_GAZEBO_ENV_SCARA4=[^;]*,'GYM_GAZEBO_ENV_SCARA4=`pwd`/../assets/worlds/scara_basic.world'," -i ~/.bashrc'
fi

if [ -z "$GYM_GAZEBO_ENV_SCARA6" ]; then
  bash -c 'echo "export GYM_GAZEBO_ENV_SCARA6="`pwd`/../assets/worlds/scara_basic.world >> ~/.bashrc'
else
  bash -c 'sed "s,GYM_GAZEBO_ENV_SCARA6=[^;]*,'GYM_GAZEBO_ENV_SCARA6=`pwd`/../assets/worlds/scara_basic.world'," -i ~/.bashrc'
fi

#copy altered urdf model
# cp -r ../assets/urdf/kobuki_urdf/urdf/ catkin_ws/src/kobuki/kobuki_description

exec bash # reload bash
