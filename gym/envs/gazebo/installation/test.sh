#!/bin/sh

#add own models path to gazebo models path
if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH="`pwd`/../assets/models >> ~/.bashrc'
  exec bash #reload bashrc
fi
