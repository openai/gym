#gym-gazebo

Extending the OpenAI Gym for robotics: a toolkit for reinforcement learning using ROS and Gazebo.

Visit [erlerobotics/gym](http://erlerobotics.com/docs/Simulation/Gym/) for more information and videos.

The following are some of the available gazebo environments for the [Turtlebot]([Soccer environment]), one of the currently supported robots:

### GazeboCircuit2TurtlebotLidar-v0

A simple circuit with straight tracks and 90 degree turns. Highly discretized LIDAR readings are used to train the Turtlebot. Scripts implementing **Q-learning** and **Sarsa** can be found in the _examples_ folder.

### GazeboCircuit2TurtlebotLidarNn-v0

A simple circuit with straight tracks and 90 degree turns. A LIDAR is used to train the Turtlebot. Scripts implementing **DQN** can be found in the _examples_ folder.

### GazeboCircuit2cTurtlebotCameraNnEnv-v0

A simple circuit with straight tracks and 90 degree turns with high contrast colors between the floor and the walls. A camera is used to train the Turtlebot. Scripts implementing **DQN** using **CNN** can be found in the _examples_ folder.


# Installation

## Table of Contents
- [Requirements](#requirements)
- [ROS Indigo](#ros-indigo)
- [Gazebo](#gazebo)
- [Gym Gazebo Pip](#gym-gazebo-pip)
- [Dependencies](#dependencies)
  - [Automatic installation](#automatic-installation)
  - [Step-by-step installation](#step-by-step-installation)
  - [Keras and Theano installation](#keras-and-theano-installation)
    - [Enablig GPU for Theano](#enablig-gpu-for-theano)
- [Troubleshooting](#troubleshooting)

## Requirements

- Ubuntu 14.04
- 2GB free space

## ROS Indigo

Install the Robot Operating System via:

**ros-indigo-desktop-full** is the only recommended installation.

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
- 4Install Gazebo:

```bash
sudo apt-get update
sudo apt-get remove .*gazebo.* && sudo apt-get update && sudo apt-get install gazebo7 libgazebo7-dev
```

## Gym Gazebo Pip

```bash
cd gym-gazebo
sudo pip install -e .
```

## Dependencies

There are two options to install dependencies: automatic installation or step-by-step installation

### Automatic installation

Install dependencies running [setup.bash](gym_gazebo/envs/installation/setup.bash). If you are going to use DQN with Keras, also install [Keras and Theano](#keras-and-theano-installation).

```bash
cd gym_gazebo/envs/installation
bash setup.bash
```
Before running a environment, load the corresponding setup script. For example, to load the Turtlebot execute:

- Turtlebot

```bash
cd gym_gazebo/envs/installation
bash turtlebot_setup.bash
```

### Step-by-step installation

**1.** Install dependencies

Install ubuntu packages
```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver hkp://pool.sks-keyservers.net --recv-key 0xB01FA116
sudo apt-get update
sudo apt-get install -y git                            \
                        mercurial                      \
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
                        gawk                           \
                        libtinyxml2-dev
```

Install python dependencies

```bash
sudo easy_install numpy
sudo pip2 install pymavlink MAVProxy catkin_pkg --upgrade
```

Install Sophus
```bash
cd gym_gazebo/envs/installation
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
Then, create the catkin workspace inside `gym_gazebo/envs/installation/` directory

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
catkin_make
# If your computer crashes, try 'catkin_make -j1' instead
```
**4.** Add GAZEBO_MODEL_PATH to your `bashrc` and load it

```bash
cd ../../assets/models
echo "export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:$(pwd)" >> ~/.bashrc
source ~/.bashrc
```

**5.** Before running a environment, load the corresponding setup script. For example, to load the Turtlebot execute:

- Turtlebot

```bash
cd gym_gazebo/envs/installation
bash turtlebot_setup.bash
```

### Keras and Theano installation
This part of the installation is required only for the environments using DQN.

```bash
# install dependencies
sudo pip install h5py
sudo apt-get install gfortran

# install Theano
git clone git://github.com/Theano/Theano.git
cd Theano/
sudo python setup.py develop

#isntall Keras
sudo pip install keras
```
dot_parser error fix:
```bash
sudo pip install --upgrade pydot
sudo pip install --upgrade pyparsing
```

#### Enablig GPU for Theano

Follow the instructions [here](http://deeplearning.net/software/theano/install.html#gpu-linux) and change $PATH instead of $CUDA_ROOT.

Working on a clean installation of Ubuntu 14.04 using CUDA 7.5.

The following flags are needed in order to execute in GPU mode, using an alias is recommeneded.
```bash
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
```