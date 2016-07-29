#!/bin/bash

if [ -z "$ROS_DISTRO" ]; then
  echo "ROS not installed. Check the installation steps: https://github.com/erlerobot/gym#installing-the-gazebo-environment"
fi

program="gazebo"
condition=$(which $program 2>/dev/null | grep -v "not found" | wc -l)
if [ $condition -eq 0 ] ; then
    echo "Gazebo is not installed. Check the installation steps: https://github.com/erlerobot/gym#installing-the-gazebo-environment"
fi

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
sudo apt-get install -y git                            \
                        libsdl-image1.2-dev            \
                        libspnav-dev                   \
                        libtbb-dev                     \
                        libtbb2                        \
                        libusb-dev libftdi-dev         \
                        pyqt4-dev-tools                \
                        python-vcstool                 \
                        ros-indigo-bfl                 \
                        python-pip
sudo easy_install numpy
sudo pip install --upgrade matplotlib

# Import and build dependencies
vcs import < ../../gazebo.repos
echo 'SET(CMAKE_CXX_FLAGS "-std=c++11")' >> kobuki_desktop/kobuki_gazebo_plugins/CMakeLists.txt
cd ..
catkin_make_isolated

#add own models path to gazebo models path
if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH="`pwd`/../../assets/models >> ~/.bashrc'
  exec bash #reload bashrc
fi
