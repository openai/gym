"""
Docs on how to do the markdown formatting:
http://docutils.sourceforge.net/docs/user/rst/quickref.html

Tool for previewing the markdown:
http://rst.ninjs.org/
"""

import os

from gym.scoreboard.registration import registry, add_task, add_group

# Discover API key from the environment. (You should never have to
# change api_base / web_base.)
'''api_key = os.environ.get('OPENAI_GYM_API_KEY')
api_base = os.environ.get('OPENAI_GYM_API_BASE', 'https://gym-api.openai.com')
web_base = os.environ.get('OPENAI_GYM_WEB_BASE', 'https://gym.openai.com')'''

# The following controls how various tasks appear on the
# scoreboard. These registrations can differ from what's registered in
# this repository.

# groups

add_group(
    id='gazebo',
    name='Gazebo',
    description='TODO.'
)

add_task(
    id='GazeboMazeTurtlebotLidar-v0',
    group='gazebo',
    summary='Obstacle avoidance in a Maze.',
)
add_task(
    id='GazeboCircuitTurtlebotLidar-v0',
    group='gazebo',
    summary='Obstacle avoidance in a Circuit.',
)
add_task(
    id='GazeboCircuit2TurtlebotLidar-v0',
    group='gazebo',
    summary='Obstacle avoidance in a Circuit 2.',
)
add_task(
    id='GazeboCircuit2TurtlebotLidarNn-v0',
    group='gazebo',
    summary='Obstacle avoidance in a Circuit 2 using continuous data.',
)
add_task(
    id='GazeboCircuit2cTurtlebotCameraNnEnv-v0',
    group='gazebo',
    summary='Obstacle avoidance in Circuit 2 with colors using camera data.',
)
add_task(
    id='GazeboRoundTurtlebotLidar-v0',
    group='gazebo',
    summary='Obstacle avoidance in a Round circuit.',
)
add_task(
    id='GazeboMazeErleRoverLidar-v0',
    group='gazebo',
    summary='Obstacle avoidance with Erle-Rover in a maze.',
)
add_task(
    id='GazeboErleCopterHover-v0',
    group='gazebo',
    summary='Hover a point with Erle-Copter',
)

registry.finalize()
