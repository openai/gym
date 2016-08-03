# Installing the Gazebo environment

## Requirements

- Ubuntu 14.0
- Gazebo7
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
git clone https://github.com/strasdat/Sophus
cd Sophus
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
# Inside installation/ folder
mkdir -p catkin_ws/src
cd catkin_ws/src
catkin_init_workspace
```
**3.** Import packages into catkin workspace and build

```bash
vcs import < ../../gazebo.repos
cd ..
catkin_make_isolated
```
**4.** Add GAZEBO_MODEL_PATH to your `bashrc`

```bash
echo "export GAZEBO_MODEL_PATH="`pwd`/../../assets/models >> ~/.bashrc
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