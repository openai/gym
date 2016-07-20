#!/bin/bash

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
                     ros-indigo-bfl                 \
                     python-pip
sudo pip install numpy --upgrade
sudo pip install pandas
# Import and build dependencies
vcs import < ../../gazebo.repos
echo 'SET(CMAKE_CXX_FLAGS "-std=c++11")' >> kobuki_desktop/kobuki_gazebo_plugins/CMakeLists.txt
cd ..
catkin_make_isolated
source devel_isolated/setup.bash

#add own models path to gazebo models path
if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH="`pwd`/../assets/models >> ~/.bashrc'
  exec bash #reload bashrc
fi

#--TURTLEBOT--#
sh ./turtlebotSetup.sh
