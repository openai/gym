# Installing the Gazebo environment

## Table of Contents
- [Requirements](#requirements)
- [ROS Indigo](#ros-indigo)
- [Gazebo](#gazebo)
- [Dependencies](#dependencies)
  - [Automatic Installation](#automatic-installation)
  - [Step-by-step installation](#step-by-step-installation)
- [Troubleshooting](#troubleshooting)

## Requirements

- Ubuntu 14.04
- 2GB free space

## ROS Indigo

Install the Robot Operating System via:

- Ubuntu: http://wiki.ros.org/indigo/Installation/Ubuntu
- Others: http://wiki.ros.org/indigo/Installation 

## Gazebo

- Setup your computer to accept software from packages.osrfoundation.org:

```bash
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
```
- Setup keys:

```bash
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
```
- Install Gazebo:

```bash
sudo apt-get update
sudo apt-get remove .*gazebo.* && sudo apt-get update && sudo apt-get install gazebo7 libgazebo7-dev
```

- Check your installation:

```bash
gazebo
```

## Dependencies

There are two options to install dependencies: automatic installation or step-by-step installation

### Automatic installation

Install dependencies running [setup.bash](installation/setup.bash).

```bash
cd gym/envs/gazebo/installation
bash setup.bash
```
Before running a environment, load the corresponding setup script:

- Turtlebot

```bash
cd gym/envs/gazebo/installation
bash turtlebot_setup.bash
```
- Erle-Rover

```bash
cd gym/envs/gazebo/installation
bash erlerover_setup.bash
```

- Erle-Copter

```bash
cd gym/envs/gazebo/installation
bash erlecopter_setup.bash
```

### Step-by-step installation

**1.** Install dependencies

Install ubuntu packages
```bash
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
                        python-pip                     \
                        g++                            \
                        ccache                         \
                        realpath                       \
                        libopencv-dev                  \
                        libtool                        \
                        automake                       \
                        autoconf                       \
                        libexpat1-dev                  \
                        ros-indigo-mavlink             \
                        ros-indigo-octomap-msgs        \
                        ros-indigo-joy                 \
                        ros-indigo-geodesy             \
                        ros-indigo-octomap-ros         \
                        ros-indigo-control-toolbox     \
                        drcsim                         \
                        gawk
```

Install python dependencies

```bash
sudo easy_install numpy
sudo pip2 install pymavlink MAVProxy catkin_pkg --upgrade
```

Install Sophus
```bash
cd gym/envs/gazebo/installation
git clone https://github.com/stonier/sophus -b indigo
cd sophus
mkdir build
cd build
cmake ..
make
sudo make install
```

Clone Ardupilot
```bash
cd ../..
mkdir apm && cd apm
git clone https://github.com/erlerobot/ardupilot -b gazebo
```

Install JSBSim
```bash
git clone git://github.com/tridge/jsbsim.git
# Additional dependencies required
sudo apt-get install libtool automake autoconf libexpat1-dev 
cd jsbsim
./autogen.sh --enable-libraries
make -j2
sudo make install
```

**2.** Create a catkin workspace

First load ROS environment variables
```bash
source /opt/ros/indigo/setup.bash
```
Then, create the catkin workspace inside `gym/envs/gazebo/installation/` directory

```bash
cd ../..
mkdir -p catkin_ws/src # Inside installation/ folder
cd catkin_ws/src
catkin_init_workspace
```
**3.** Import packages into catkin workspace and build

```bash
cd ../../catkin_ws/src/
vcs import < ../../gazebo.repos
cd ..
catkin_make --pkg mav_msgs
```
**4.** Add GAZEBO_MODEL_PATH to your `bashrc` and load it

```bash
cd ../../assets/models
echo "export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:$(pwd)" >> ~/.bashrc
source ~/.bashrc
```

**5.** Before running a environment, load the corresponding setup script:

- Turtlebot

```bash
cd gym/envs/gazebo/installation
bash turtlebot_setup.bash
```
- Erle-Rover

```bash
cd gym/envs/gazebo/installation
bash erlerover_setup.bash
```

- Erle-Copter

```bash
cd gym/envs/gazebo/installation
bash erlecopter_setup.bash
```

## Troubleshooting

### I can't install drcsim from Ubuntu packages

If you can't install drcsim using Ubuntu packages, you might need to install drcsim **from source** instead. So this is how to do it (instructions are based on [these](http://gazebosim.org/tutorials?tut=drcsim_install#UbuntuandROSIndigo)) :

- Configure your system to install packages from ROS Indigo. E.g., on trusty:
```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu trusty main" > /etc/apt/sources.list.d/ros-latest.list'
wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
```
- Install compile-time prerequisites:
```bash
sudo apt-get update

# Install osrf-common's dependencies
sudo apt-get install -y cmake               \
                        debhelper           \
                        ros-indigo-ros      \
                        ros-indigo-ros-comm

# Install sandia-hand's dependencies
sudo apt-get install -y ros-indigo-xacro        \
                        ros-indigo-ros          \
                        ros-indigo-image-common \
                        ros-indigo-ros-comm     \
                        ros-indigo-common-msgs  \
                        libboost-dev            \
                        avr-libc                \
                        gcc-avr                 \
                        libqt4-dev

# Install gazebo-ros-pkgs
sudo apt-get install -y libtinyxml-dev                 \
                        libtinyxml2-dev                \
                        ros-indigo-vision-opencv       \
                        ros-indigo-angles              \
                        ros-indigo-cv-bridge           \
                        ros-indigo-driver-base         \
                        ros-indigo-dynamic-reconfigure \
                        ros-indigo-geometry-msgs       \
                        ros-indigo-image-transport     \
                        ros-indigo-message-generation  \
                        ros-indigo-nav-msgs            \
                        ros-indigo-nodelet             \
                        ros-indigo-pcl-conversions     \
                        ros-indigo-pcl-ros             \
                        ros-indigo-polled-camera       \
                        ros-indigo-rosconsole          \
                        ros-indigo-rosgraph-msgs       \
                        ros-indigo-sensor-msgs         \
                        ros-indigo-trajectory-msgs     \
                        ros-indigo-urdf                \
                        ros-indigo-dynamic-reconfigure \
                        ros-indigo-rosgraph-msgs       \
                        ros-indigo-tf                  \
                        ros-indigo-cmake-modules

# Install drcsim's dependencies
sudo apt-get install -y cmake debhelper                          \
                     ros-indigo-std-msgs ros-indigo-common-msgs  \
                     ros-indigo-image-common ros-indigo-geometry \
                     ros-indigo-ros-control                      \
                     ros-indigo-geometry-experimental            \
                     ros-indigo-robot-state-publisher            \
                     ros-indigo-image-pipeline                   \
                     ros-indigo-image-transport-plugins          \
                     ros-indigo-compressed-depth-image-transport \
                     ros-indigo-compressed-image-transport       \
                     ros-indigo-theora-image-transport           \
                     ros-indigo-laser-assembler
```
- Create the catkin workspace. Default branches of ros gazebo plugins, osrf-common, sandia-hand and drcsim will be included into the workspace.
```bash
 # Setup the workspace
 mkdir -p /tmp/ws/src
 cd /tmp/ws/src

 # Download needed software
 git clone https://github.com/ros-simulation/gazebo_ros_pkgs.git
 hg clone https://bitbucket.org/osrf/osrf-common
 hg clone https://bitbucket.org/osrf/sandia-hand
 hg clone https://bitbucket.org/osrf/drcsim

 # Change to the *indigo* branch in gazebo_ros_pkgs
 cd gazebo_ros_pkgs
 git checkout indigo-devel
 cd ..

 # Source ros distro's setup.bash
 source /opt/ros/indigo/setup.bash

 # Build and install into workspace
 cd /tmp/ws
 catkin_make install -DCMAKE_INSTALL_PREFIX=/opt/ros/indigo
```

**NOTE:** you might need sudo privileges for the last command, so first run
```bash
sudo -i
source /opt/ros/indigo/setup.bash
cd /tmp/ws
```