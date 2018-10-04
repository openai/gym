import gym
import rospy
import roslaunch
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from gym.utils import seeding
import copy
import math

import numpy as np

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from osrf_gear.msg import LogicalCameraImage
from osrf_gear.msg import VacuumGripperState # /ariac/gripper/state

from osrf_gear.srv import VacuumGripperControl

# from custom baselines repository
from baselines.agent.utility.general_utils import forward_kinematics, get_ee_points, rotation_from_matrix, \
    get_rotation_matrix,quaternion_from_matrix # For getting points and velocities.
from baselines.agent.scara_arm.tree_urdf import treeFromFile # For KDL Jacobians
from PyKDL import Jacobian, Chain, ChainJntToJacSolver, JntArray # For KDL Jacobians

class MSG_INVALID_JOINT_NAMES_DIFFER(Exception):
    """Error object exclusively raised by _process_observations."""
    pass

class ARIACPickv0Env(gazebo_env.GazeboEnv):
    def __init__(self):

        self.slowness = 1
        self.slowness_unit = 'sec'
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "ARIACPick_v0.launch")

        JOINT_PUBLISHER = '/scara_controller/command'
        JOINT_SUBSCRIBER = '/scara_controller/state'

        # Set constants for joints
        SHOULDER_PAN_JOINT = 'shoulder_pan_joint'
        SHOULDER_LIFT_JOINT = 'shoulder_lift_joint'
        ELBOW_JOINT = 'elbow_joint'
        WRIST_1_JOINT = 'wrist_1_joint'
        WRIST_2_JOINT = 'wrist_2_joint'
        WRIST_3_JOINT = 'wrist_3_joint'

        # Set constants for links
        BASE = 'base'
        BASE_LINK = 'base_link'
        SHOULDER_LINK = 'shoulder_link'
        UPPER_ARM_LINK = 'upper_arm_link'
        FOREARM_LINK = 'forearm_link'
        WRIST_1_LINK = 'wrist_1_link'
        WRIST_2_LINK = 'wrist_2_link'
        WRIST_3_LINK = 'wrist_3_link'
        EE_LINK = 'ee_link'
        JOINT_ORDER = [SHOULDER_PAN_JOINT, SHOULDER_LIFT_JOINT, ELBOW_JOINT,
                           WRIST_1_JOINT, WRIST_2_JOINT, WRIST_3_JOINT]
        # JOINT_ORDER= ['linear_arm_actuator_joint', 'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        LINK_NAMES = [BASE, BASE_LINK, SHOULDER_LINK, UPPER_ARM_LINK, FOREARM_LINK,
                      WRIST_1_LINK, WRIST_2_LINK, WRIST_3_LINK, EE_LINK]

        LINK_NAMES2 = [ BASE,
                        BASE_LINK,
                        SHOULDER_LINK,
                        UPPER_ARM_LINK,
                        FOREARM_LINK,
                        WRIST_1_LINK,
                        WRIST_2_LINK,
                        WRIST_3_LINK,
                        EE_LINK
                      ]
        EE_POS_TGT = np.asmatrix([0.3305805, -0.1326121, 0.3746]) # center of the H
        EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        EE_POINTS = np.asmatrix([[0, 0, 0]])
        EE_VELOCITIES = np.asmatrix([[0, 0, 0]])

        INITIAL_JOINTS = np.array([0., 0., 0.])
        STEP_COUNT = 2  # Typically 100.

        reset_condition = {
            'initial_positions': INITIAL_JOINTS,
             'initial_velocities': []
        }
        #############################

        # TODO: fix this and make it relative
        # Set the path of the corresponding URDF file from "assets"
        URDF_PATH = "/home/erle/ros_scara/ur10.urdf"

        m_joint_order = copy.deepcopy(JOINT_ORDER)
        m_link_names = copy.deepcopy(LINK_NAMES)
        m_joint_publishers = copy.deepcopy(JOINT_PUBLISHER)
        m_joint_subscribers = copy.deepcopy(JOINT_SUBSCRIBER)
        ee_pos_tgt = EE_POS_TGT
        ee_rot_tgt = EE_ROT_TGT
        # Initialize target end effector position
        ee_tgt = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T)

        self.realgoal = ee_tgt

        self.environment = {
            'T': STEP_COUNT,
            'ee_points_tgt': ee_tgt,
            'joint_order': m_joint_order,
            'link_names': m_link_names,
            # 'slowness': slowness,
            'reset_conditions': reset_condition,
            'tree_path': URDF_PATH,
            'joint_publisher': m_joint_publishers,
            'joint_subscriber': m_joint_subscribers,
            'end_effector_points': EE_POINTS,
            'end_effector_velocities': EE_VELOCITIES,
            # 'num_samples': SAMPLE_COUNT,
        }

        rospy.wait_for_service('/ariac/gripper/control')
        self.vacuum_gripper_control = rospy.ServiceProxy('/ariac/gripper/control', VacuumGripperControl)

        self._pub = rospy.Publisher('/ariac/arm/command', JointTrajectory)

        # Initialize a tree structure from the robot urdf.
        #   note that the xacro of the urdf is updated by hand.
        # The urdf must be compiled.
        _, self.ur_tree = treeFromFile(self.environment['tree_path'])
        # Retrieve a chain structure between the base and the start of the end effector.
        self.scara_chain = self.ur_tree.getChain(self.environment['link_names'][0], self.environment['link_names'][-1])
        # print("nr of jnts: ", self.scara_chain.getNrOfJoints())
        # Initialize a KDL Jacobian solver from the chain.
        self.jac_solver = ChainJntToJacSolver(self.scara_chain)
        #print(self.jac_solver)
        self._observations_stale = [False for _ in range(1)]
        #print("after observations stale")
        self._currently_resetting = [False for _ in range(1)]
        self.reset_joint_angles = [None for _ in range(1)]

        # TODO review with Risto, we might need the first observation for calling step()
        # # taken from mujoco in OpenAi how to initialize observation space and action space.
        # observation, _reward, done, _info = self.step(np.zeros(self.scara_chain.getNrOfJoints()))
        # assert not done
        # self.obs_dim = observation.size
        self.obs_dim = self.scara_chain.getNrOfJoints() + 6 # hardcode it for now
        # # print(observation, _reward)

        # # Here idially we should find the control range of the robot. Unfortunatelly in ROS/KDL there is nothing like this.
        # # I have tested this with the mujoco enviroment and the output is always same low[-1.,-1.], high[1.,1.]
        # #bounds = self.model.actuator_ctrlrange.copy()
        low = -np.pi/2.0 * np.ones(self.scara_chain.getNrOfJoints())
        high = np.pi/2.0 * np.ones(self.scara_chain.getNrOfJoints())
        # print("Action spaces: ", low, high)
        self.action_space = spaces.Box(low, high)
        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        # self.action_space = spaces.Discrete(3) #F,L,R
        # self.reward_range = (-np.inf, np.inf)

        # Gazebo specific services to start/stop its behavior and
        # facilitate the overall RL environment
        # self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        # self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        # self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # Seed the environment
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        state = []
        reward = []
        done = False

        data_arm_state = None
        while data_arm_state is None:
            try:
                data_arm_state = rospy.wait_for_message('/ariac/arm/state', JointTrajectoryControllerState, timeout=5)
            except:
                pass

        data_arm_state2 = JointTrajectoryControllerState()
        data_arm_state2.joint_names = data_arm_state.joint_names[1:]
        data_arm_state2.desired.positions  = data_arm_state.desired.positions[1:]
        data_arm_state2.desired.velocities = data_arm_state.desired.velocities[1:]
        data_arm_state2.desired.accelerations = data_arm_state.desired.accelerations[1:]
        data_arm_state2.actual.positions = data_arm_state.actual.positions[1:]
        data_arm_state2.actual.velocities = data_arm_state.actual.velocities[1:]
        data_arm_state2.actual.accelerations = data_arm_state.actual.accelerations[1:]
        data_arm_state2.error.positions = data_arm_state.error.positions[1:]
        data_arm_state2.error.velocities = data_arm_state.error.velocities[1:]
        data_arm_state2.error.accelerations = data_arm_state.error.accelerations[1:]
        # Collect the end effector points and velocities in
        # cartesian coordinates for the process_observationsstate.
        # Collect the present joint angles and velocities from ROS for the state.
        last_observations = self.process_observations(data_arm_state2, self.environment)
        # # # Get Jacobians from present joint angles and KDL trees
        # # # The Jacobians consist of a 6x6 matrix getting its from from
        # # # (# joint angles) x (len[x, y, z] + len[roll, pitch, yaw])
        ee_link_jacobians = self.get_jacobians(last_observations)
        if self.environment['link_names'][-1] is None:
            print("End link is empty!!")
            return None
        else:
            # print(self.environment['link_names'][-1])
            trans, rot = forward_kinematics(self.scara_chain,
                                        self.environment['link_names'],
                                        last_observations[:self.scara_chain.getNrOfJoints()],
                                        base_link=self.environment['link_names'][0],
                                        end_link=self.environment['link_names'][-1])
            # #
            rotation_matrix = np.eye(4)
            rotation_matrix[:3, :3] = rot
            rotation_matrix[:3, 3] = trans
            # angle, dir, _ = rotation_from_matrix(rotation_matrix)
            # #
            # current_quaternion = np.array([angle]+dir.tolist())#

            # I need this calculations for the new reward function, need to send them back to the run scara or calculate them here
            current_quaternion = quaternion_from_matrix(rotation_matrix)
            current_ee_tgt = np.ndarray.flatten(get_ee_points(self.environment['end_effector_points'],
                                                              trans,
                                                              rot).T)
            ee_points = current_ee_tgt - self.realgoal#self.environment['ee_points_tgt']
            ee_points_jac_trans, _ = self.get_ee_points_jacobians(ee_link_jacobians,
                                                                   self.environment['end_effector_points'],
                                                                   rot)
            ee_velocities = self.get_ee_points_velocities(ee_link_jacobians,
                                                           self.environment['end_effector_points'],
                                                           rot,
                                                           last_observations)

            # Concatenate the information that defines the robot state
            # vector, typically denoted asrobot_id 'x'.
            state = np.r_[np.reshape(last_observations, -1),
                          np.reshape(ee_points, -1),
                          np.reshape(ee_velocities, -1),]

        self._pub.publish(self.get_trajectory_message([0.018231524968342513,
                                                       3.2196073949389703 + np.random.rand(1)[0] - .5,
                                                      -0.6542426695478509 + np.random.rand(1)[0] - .5,
                                                       1.7401498071871018 + np.random.rand(1)[0] - .5,
                                                       3.7354939354077596 + np.random.rand(1)[0] - .5,
                                                      -1.510707301072792 + np.random.rand(1)[0] - .5,
                                                       0.00014565660628829136 + np.random.rand(1)[0] - .5]))

        data_vacuum_state = None
        while data_vacuum_state is None:
            try:
                data_vacuum_state = rospy.wait_for_message('/ariac/gripper/state', VacuumGripperState, timeout=5)
            except:
                pass

        data_camera = None
        while data_camera is None:
            try:
                data_camera = rospy.wait_for_message('/ariac/logical_camera_1', LogicalCameraImage, timeout=5)
            except:
                pass

        try:
            self.vacuum_gripper_control(enable=bool(state[0]))
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        state = [data_arm_state, data_camera, data_vacuum_state]


        return state, reward, done, {}

    def reset(self):

        data_arm_state = None
        while data_arm_state is None:
            try:
                data_arm_state = rospy.wait_for_message('/ariac/arm/state', JointTrajectoryControllerState, timeout=5)
            except:
                pass
        data_camera = None
        while data_camera is None:
            try:
                data_camera = rospy.wait_for_message('/ariac/logical_camera_1', LogicalCameraImage, timeout=5)
            except:
                pass

        state = [data_arm_state, data_camera]

        return state


    def get_trajectory_message(self, action, robot_id=0):
        """
        Helper function.
        Wraps an action vector of joint angles into a JointTrajectory message.
        The velocities, accelerations, and effort do not control the arm motion
        """
        # Set up a trajectory message to publish.
        action_msg = JointTrajectory()
        action_msg.joint_names =  copy.deepcopy(self.environment['joint_order'])
        action_msg.joint_names.insert(0, 'linear_arm_actuator_joint')
        # Create a point to tell the robot to move to.
        target = JointTrajectoryPoint()
        action_float = [float(i) for i in action]
        target.positions = action_float
        # These times determine the speed at which the robot moves:
        # it tries to reach the specified target position in 'slowness' time.
        if (self.slowness_unit == 'sec') or (self.slowness_unit is None):
            target.time_from_start.secs = self.slowness
        elif (self.slowness_unit == 'nsec'):
            target.time_from_start.nsecs = self.slowness
        else:
            print("Unrecognized unit. Please use sec or nsec.")
        # target.time_from_start.nsecs = self.environment['slowness']
        # Package the single point into a trajectory of points with length 1.
        action_msg.points = [target]
        return action_msg

    def process_observations(self, message, agent, robot_id=0):
        """
        Helper fuinction to convert a ROS message to joint angles and velocities.
        Check for and handle the case where a message is either malformed
        or contains joint values in an order different from that expected observation_callback
        in hyperparams['joint_order']
        """
        if not message:
            print("Message is empty");
            # return None
        else:
            # # Check if joint values are in the expected order and size.
            if message.joint_names != agent['joint_order']:
                # Check that the message is of same size as the expected message.
                if len(message.joint_names) != len(agent['joint_order']):
                    raise MSG_INVALID_JOINT_NAMES_DIFFER

                # Check that all the expected joint values are present in a message.
                if not all(map(lambda x,y: x in y, message.joint_names,
                    [self._valid_joint_set[robot_id] for _ in range(len(message.joint_names))])):
                    raise MSG_INVALID_JOINT_NAMES_DIFFER
                    print("Joints differ")
            return np.array(message.actual.positions) # + message.actual.velocities

    def get_jacobians(self, state, robot_id=0):
        """
        Produce a Jacobian from the urdf that maps from joint angles to x, y, z.
        This makes a 6x6 matrix from 6 joint angles to x, y, z and 3 angles.
        The angles are roll, pitch, and yaw (not Euler angles) and are not needed.
        Returns a repackaged Jacobian that is 3x6.
        """
        # Initialize a Jacobian for self.scara_chain.getNrOfJoints() joint angles by 3 cartesian coords and 3 orientation angles
        jacobian = Jacobian(self.scara_chain.getNrOfJoints())
        # Initialize a joint array for the present self.scara_chain.getNrOfJoints() joint angles.
        angles = JntArray(self.scara_chain.getNrOfJoints())
        # Construct the joint array from the most recent joint angles.
        for i in range(self.scara_chain.getNrOfJoints()):
            angles[i] = state[i]
        # Update the jacobian by solving for the given angles.observation_callback
        self.jac_solver.JntToJac(angles, jacobian)
        # Initialize a numpy array to store the Jacobian.
        J = np.array([[jacobian[i, j] for j in range(jacobian.columns())] for i in range(jacobian.rows())])
        # Only want the cartesian position, not Roll, Pitch, Yaw (RPY) Angles
        ee_jacobians = J
        return ee_jacobians

    def get_ee_points_jacobians(self, ref_jacobian, ee_points, ref_rot):
        """
        Get the jacobians of the points on a link given the jacobian for that link's origin
        :param ref_jacobian: 6 x 6 numpy array, jacobian for the link's origin
        :param ee_points: N x 3 numpy array, points' coordinates on the link's coordinate system
        :param ref_rot: 3 x 3 numpy array, rotational matrix for the link's coordinate system
        :return: 3N x 6 Jac_trans, each 3 x 6 numpy array is the Jacobian[:3, :] for that point
                 3N x 6 Jac_rot, each 3 x 6 numpy array is the Jacobian[3:, :] for that point
        """
        ee_points = np.asarray(ee_points)
        ref_jacobians_trans = ref_jacobian[:3, :]
        ref_jacobians_rot = ref_jacobian[3:, :]
        end_effector_points_rot = np.expand_dims(ref_rot.dot(ee_points.T).T, axis=1)
        ee_points_jac_trans = np.tile(ref_jacobians_trans, (ee_points.shape[0], 1)) + \
                                        np.cross(ref_jacobians_rot.T, end_effector_points_rot).transpose(
                                            (0, 2, 1)).reshape(-1, self.scara_chain.getNrOfJoints())
        ee_points_jac_rot = np.tile(ref_jacobians_rot, (ee_points.shape[0], 1))
        return ee_points_jac_trans, ee_points_jac_rot

    def get_ee_points_velocities(self, ref_jacobian, ee_points, ref_rot, joint_velocities):
        """
        Get the velocities of the points on a link
        :param ref_jacobian: 6 x 6 numpy array, jacobian for the link's origin
        :param ee_points: N x 3 numpy array, points' coordinates on the link's coordinate system
        :param ref_rot: 3 x 3 numpy array, rotational matrix for the link's coordinate system
        :param joint_velocities: 1 x 6 numpy array, joint velocities
        :return: 3N numpy array, velocities of each point
        """
        ref_jacobians_trans = ref_jacobian[:3, :]
        ref_jacobians_rot = ref_jacobian[3:, :]
        ee_velocities_trans = np.dot(ref_jacobians_trans, joint_velocities)
        ee_velocities_rot = np.dot(ref_jacobians_rot, joint_velocities)
        ee_velocities = ee_velocities_trans + np.cross(ee_velocities_rot.reshape(1, 3),
                                                       ref_rot.dot(ee_points.T).T)
        return ee_velocities.reshape(-1)
