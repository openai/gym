# A Dockerfile for the gym-gazebo environment
FROM ubuntu:16.04

#--------------------
# General setup
#--------------------

# Get the dependencies
RUN apt-get update \
    && apt-get install -y xorg-dev \
    libgl1-mesa-dev \
    xvfb \
    libxinerama1 \
    libxcursor1 \
    unzip \
    libglu1-mesa \
    libav-tools \
    python3 \
    python3-pip \
    # python3-numpy \
    # python3-scipy \
    # python3-pyglet \
    python3-setuptools \
    libpq-dev \
    libjpeg-dev \
    wget \
    curl \
    cmake \
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local/gym

#--------------------
# Install gym
#--------------------

# # Clone the official gym
# RUN git clone https://github.com/openai/gym
#
# # Install the gym's requirements
# RUN pip install -r gym/requirements.txt
#
# # Install the gym
# RUN ls -l
# RUN pip install -e gym/

# Install from pip
RUN pip3 install gym

# Checks
#RUN python --version
#RUN python -c "import gym"

# Debug
#RUN ls -l /usr/local/gym
#RUN ls -l /usr/local/gym/gym-gazebo
#RUN ls -l /usr/local/gym/gym

# #--------------------
# # Install Gazebo
# #--------------------
RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable xenial main" > /etc/apt/sources.list.d/gazebo-stable.list'

RUN wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add -

RUN apt-get update
RUN apt-get install gazebo8 -y
# RUN apt-get install -y libglib2.0-dev libgts-dev libgts-dev
RUN apt-get install -y libgazebo8-dev

# setup environment
EXPOSE 11345


#--------------------
# Install ROS
#--------------------

# RUN apt-get install -y locales-all
# # setup environment
# RUN locale-gen en_US.UTF-8
# ENV LANG en_US.UTF-8

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# setup keys
RUN apt-key adv --keyserver ha.pool.sks-keyservers.net --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list
# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
#
# # install ros packages
ENV ROS_DISTRO kinetic
RUN apt-get install cmake gcc g++

# # Install from repositories (for Python 2.7)
#----------------------------
# install bootstrap tools
# RUN apt-get update && apt-get install --no-install-recommends -y \
#     python-rosdep \
#     python-rosinstall \
#     python-vcstools \
#     && rm -rf /var/lib/apt/lists/*
# RUN apt-get update && apt-get install -y \
#       ros-kinetic-ros-base && rm -rf /var/lib/apt/lists/*
#
# RUN apt-get install ros-kinetic-*
# # Install additional dependencies
# RUN apt-get install -y ros-kinetic-cv-bridge
# RUN apt-get install -y ros-kinetic-robot-state-publisher ros-kinetic-control-msgs

# Install from sources
#----------------------------
# RUN apt-get install python3-rosdep python3-rosinstall-generator python3-wstool \
#           python3-rosinstall build-essential
# or alternatively,
RUN apt-get update && apt-get install libboost-all-dev -y

RUN pip3 install --upgrade pip
RUN pip3 install -U rosdep rosinstall_generator wstool rosinstall
RUN pip3 install rospkg catkin_pkg empy

# bootstrap rosdep
RUN rosdep init \
    && rosdep update

RUN mkdir ~/ros_catkin_ws
# Create package set
RUN cd ~/ros_catkin_ws && rosinstall_generator ros_comm --rosdistro kinetic \
            --deps --wet-only --tar > kinetic-ros_comm-wet.rosinstall
# Fetch packages
RUN cd ~/ros_catkin_ws && wstool init -j1 src kinetic-ros_comm-wet.rosinstall
# # Solve dependencies
# RUN cd ~/ros_catkin_ws && rosdep install --from-paths src --ignore-src --rosdistro kinetic -y

# Create symbolic link for the compilation
RUN cd /usr/bin && ln -sf python3 python


# Install console_bridge from packages
RUN apt-get install libconsole-bridge-dev -y
# # Compile/install console_bridge as a library
# RUN git clone git://github.com/ros/console_bridge.git
# RUN cd console_bridge && cmake . && make
# RUN cd console_bridge && make install

RUN apt-get install -y libtinyxml-dev liblz4-dev libbz2-dev liburdfdom-dev libpoco-dev \
              libtinyxml2-dev

# Compile the basic ROS packages, optimize docker production
RUN cd ~/ros_catkin_ws && ./src/catkin/bin/catkin_make_isolated -DPYTHON_VERSION=3.5 --install -DCMAKE_BUILD_TYPE=Release

# Add a few packages and dependencies by hand
# RUN cd ~/ros_catkin_ws/src && git clone https://github.com/ros/console_bridge
RUN cd ~/ros_catkin_ws/src && git clone https://github.com/ros-controls/control_toolbox
RUN cd ~/ros_catkin_ws/src && git clone https://github.com/ros-controls/realtime_tools
RUN cd ~/ros_catkin_ws/src && git clone https://github.com/ros/actionlib
RUN cd ~/ros_catkin_ws/src && git clone https://github.com/ros/pluginlib
RUN cd ~/ros_catkin_ws/src && git clone https://github.com/ros/class_loader
RUN cd ~/ros_catkin_ws/src && git clone https://github.com/ros/urdf
RUN cd ~/ros_catkin_ws/src && git clone https://github.com/ros-simulation/gazebo_ros_pkgs
RUN cd ~/ros_catkin_ws/src && git clone https://github.com/ros/common_msgs
RUN cd ~/ros_catkin_ws/src && git clone https://github.com/ros-controls/control_msgs
RUN cd ~/ros_catkin_ws/src && git clone https://github.com/vmayoral/dynamic_reconfigure


# # #--------------------
# # # Install Gazebo
# # #--------------------
# RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable xenial main" > /etc/apt/sources.list.d/gazebo-stable.list'
#
# RUN wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add -
#
# RUN apt-get update
# RUN apt-get install gazebo8 -y
# # RUN apt-get install -y libglib2.0-dev libgts-dev libgts-dev
# RUN apt-get install -y libgazebo8-dev
#
# # setup environment
# EXPOSE 11345

# #--------------------
# # Follow up with the ROS intallation, splited in this funny way to optimize docker's performance
# #--------------------

# Compile the again the workspace
RUN cd ~/ros_catkin_ws && ./src/catkin/bin/catkin_make_isolated -DPYTHON_VERSION=3.5 \
        --install -DCMAKE_BUILD_TYPE=Release -DCATKIN_ENABLE_TESTING=0


# Debug
# RUN ls -l /opt/ros

# upgrade pip
#RUN apt-get install python3-pyqt4


# #--------------------
# # Install Sophus
# #--------------------
# RUN git clone https://github.com/stonier/sophus -b indigo && \
#     cd sophus && mkdir build && cd build && cmake .. && make
# # RUN ls -l
# RUN cd sophus/build && make install
# #RUN echo "## Sophus installed ##\n"
#
# #--------------------
# # Install OpenCV
# #--------------------
# # From sources
# RUN git clone https://github.com/opencv/opencv
# RUN cd opencv && mkdir build && cd build && cmake .. && make
# RUN cd opencv/build && make install
#
# # # FROM pip
# # RUN pip3 install opencv-python
# #RUN cd /usr/local/gym
#
# # More dependencies
# RUN pip3 install h5py
# RUN apt-get install -y python3-skimage \
#     bash-completion \
#     python3-defusedxml
#
# RUN pip3 install baselines
#
# #--------------------
# # Copy the code
# #--------------------
# # this invalidates the cache
# RUN mkdir gym-gazebo
# COPY . /usr/local/gym/gym-gazebo
#
# # #--------------------
# # # Install deep learning toolkits
# # #--------------------
# # # install dependencies
# # RUN pip install h5py
# # RUN apt-get install gfortran -y
# #
# # # install sript specific dependencies (temporal)
# # RUN apt-get install python-skimage -y
# #
# # # install Theano
# # #RUN git clone git://github.com/Theano/Theano.git
# # #RUN cd Theano/ && python setup.py develop
# # RUN pip install Theano
# #
# # # install Keras
# # RUN pip install keras
#
# #--------------------
# # Install gym-gazebo
# #--------------------
#
# RUN cd gym-gazebo && pip install -e .
#
# # # old method
# # # install dependencies
# # RUN cd /usr/local/gym/gym-gazebo/gym_gazebo/envs/installation && bash setup.bash
#
# #WORKDIR /root
# #ENTRYPOINT ["/usr/local/gym/bin/docker_entrypoint"]
#
# # setup entrypoint
# #RUN ls /usr/local/gym/gym-gazebo/
# #RUN ls ./gym-gazebo
# #COPY /usr/local/gym/gym-gazebo/entrypoint.sh /
#
# #--------------------
# # Entry point
# #--------------------
#
# COPY entrypoint.sh /
#
# ENTRYPOINT ["/entrypoint.sh"]
# CMD ["bash"]
