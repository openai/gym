#Environment Setup

###Install ROS-Indigo

Ubuntu: http://wiki.ros.org/indigo/Installation/Ubuntu

Others: http://wiki.ros.org/indigo/Installation 

###Install Gazebo6

####Step-by-step Install [Ubuntu]
1. Setup your computer to accept software from packages.osrfoundation.org.
'''
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
'''
2. Setup keys.
'''
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
'''
3. Install Gazebo.
'''
sudo apt-get update
sudo apt-get remove .*gazebo.* && sudo apt-get update && sudo apt-get install gazebo6 libgazebo6-dev
'''
4. Check your installation.
'''
gazebo
'''
