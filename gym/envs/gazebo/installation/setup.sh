#!/bin/sh

source /opt/ros/indigo/setup.bash

# Create catkin_ws
ws="catkin_ws"
if [ -d $ws ]; then
  echo "Error: catkin_ws directory already exists" 1>&2
  exit 1
fi
src=$ws"/src"
mkdir -p $src
cd $src
catkin_init_workspace
cd ..
catkin_make_isolated
cd src

# Install dependencies
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver hkp://pool.sks-keyservers.net --recv-key 0xB01FA116
sudo apt-get update
sudo apt-get install git                            \
                     libsdl-image1.2-dev            \
                     libspnav-dev                   \
                     libtbb-dev                     \
                     libtbb2                        \
                     libusb-dev libftdi-dev         \
                     pyqt4-dev-tools                \
                     python-vcstool                 \
                     ros-indigo-bfl
sudo apt-get python-pip
sudo pip install numpy --upgrade
sudo pip install pandas
# Import and build dependencies
vcs import < ../../gazebo.repos
echo 'SET(CMAKE_CXX_FLAGS "-std=c++11")' >> kobuki_desktop/kobuki_gazebo_plugins/CMakeLists.txt
cd ..
source devel_isolated/setup.bash
catkin_make_isolated

#add own models path to gazebo models path
if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH="`pwd`/../assets/models >> ~/.bashrc'
  exec bash #reload bashrc
fi


#---------------TURTLEBOT------------------#
#Load turtlebot variables. Temporal solution
chmod +x $src/turtlebot_simulator/turtlebot_gazebo/env-hooks/25.turtlebot-gazebo.sh.em
bash $src/turtlebot_simulator/turtlebot_gazebo/env-hooks/25.turtlebot-gazebo.sh.em

#add turtlebot launch environment variable
if [ -z "$GYM_GAZEBO_WORLD_MAZE" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_MAZE="`pwd`/../assets/worlds/maze.world >> ~/.bashrc'
  exec bash #reload bashrc
fi

#copy altered urdf model
bash -c "cp -r ../assets/urdf/kobuki_urdf/urdf/ catkin_ws/src/kobuki/kobuki_description"

#exit 0
