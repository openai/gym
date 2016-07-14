#Gazebo conventions

##Environment naming
Gazebo\<World\>\<Robot\>\<Sensors\>.v\<version\>

* World: Descriptive name of the world or main model starting with capital letter.

* Robot: Robot name starting with capital letter.

* Sensors: Sensor names used by the Robot, each starting with capital letter. Concatenations made using '-' character.

* version: Integer starting with 0.

Examples:

GazeboTurtlebotLidar_v0, GazeboTurtlebotLidar_v1, GazeboKobukiLidar-Camera-DepthSensor_v0
