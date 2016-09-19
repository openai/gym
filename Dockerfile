# A Dockerfile for the gym-gazebo environment
FROM ubuntu:14.04

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
    python-numpy \
    python-scipy \
    python-pyglet \
    python-setuptools \
    libpq-dev \
    libjpeg-dev \
    wget \
    curl \
    cmake \
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && easy_install pip

WORKDIR /usr/local/gym

# Copy the code
RUN mkdir gym-gazebo 
COPY . /usr/local/gym/gym-gazebo

#--------------------
# Install gym
#--------------------
# Clone the official gym
RUN git clone https://github.com/openai/gym

# Install the gym's requirements
RUN pip install -r gym/requirements.txt

# Install the gym
RUN ls -l
RUN pip install -e gym/

# Checks
#RUN python --version
#RUN python -c "import gym"

# Debug
#RUN ls -l /usr/local/gym
#RUN ls -l /usr/local/gym/gym-gazebo
#RUN ls -l /usr/local/gym/gym

#--------------------
# Install ROS
#--------------------
# setup environment
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8

# setup keys
RUN apt-key adv --keyserver ha.pool.sks-keyservers.net --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu trusty main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init \
    && rosdep update

# install ros packages
ENV ROS_DISTRO indigo
RUN apt-get update && apt-get install -y \
    ros-indigo-ros-core=1.1.4-0* \
    && rm -rf /var/lib/apt/lists/*
#    ros-indigo-desktop-full

#--------------------
# Install Gazebo
#--------------------
RUN sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'

RUN wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

RUN sudo apt-get update
RUN sudo apt-get install gazebo7 libgazebo7-dev -y

# setup environment
EXPOSE 11345

# Install additional dependencies
RUN apt-get install -y ros-indigo-cv-bridge
RUN apt-get install -y ros-indigo-robot-state-publisher


#--------------------
# Install deep learning toolkits
#--------------------
# install dependencies
RUN sudo pip install h5py
RUN sudo apt-get install gfortran -y

# install sript specific dependencies (temporal)
RUN sudo apt-get install python-skimage -y

# install Theano
#RUN git clone git://github.com/Theano/Theano.git
#RUN cd Theano/ && sudo python setup.py develop
RUN sudo pip install Theano

# install Keras
RUN sudo pip install keras

#--------------------
# Install gym-gazebo
#--------------------

RUN cd gym-gazebo && sudo pip install -e .

# install dependencies
RUN cd /usr/local/gym/gym-gazebo/gym_gazebo/envs/installation && bash setup.bash

#WORKDIR /root
#ENTRYPOINT ["/usr/local/gym/bin/docker_entrypoint"]

# setup entrypoint
#RUN ls /usr/local/gym/gym-gazebo/
#RUN ls ./gym-gazebo
#COPY /usr/local/gym/gym-gazebo/entrypoint.sh /

#--------------------
# Entry point
#--------------------

COPY entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
