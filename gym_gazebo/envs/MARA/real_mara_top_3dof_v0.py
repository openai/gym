import gym
import rospy
import roslaunch
import time
import random
import numpy as np
from gym import utils, spaces
from gym_gazebo.envs import real_env
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetLinkState, SetModelState
from gazebo_msgs.msg import LinkState, ModelState
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gym.utils import seeding
import copy
import rospkg
import threading # Used for time locks to synchronize position data.

from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge, CvBridgeError

import threading # Used for time locks to synchronize position data.
# since tf is really pain in the asss to work with python3 I use something different:
import quaternion as quat

from pykalman import KalmanFilter

import cv2
import time

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from baselines.agent.scara_arm.tree_urdf import treeFromFile # For KDL Jacobians
from PyKDL import Jacobian, Chain, ChainJntToJacSolver, JntArray # For KDL Jacobians

# from custom baselines repository
from baselines.agent.utility.general_utils import forward_kinematics, get_ee_points, rotation_from_matrix, \
    get_rotation_matrix,quaternion_from_matrix # For getting points and velocities.

class MSG_INVALID_JOINT_NAMES_DIFFER(Exception):
    """Error object exclusively raised by _process_observations."""
    pass

from collections import deque

class RingBuffer(deque):
    """
    inherits deque, pops the oldest data to make room
    for the newest data when size is reached
    """
    def __init__(self, size):
        deque.__init__(self)
        self.size = size

    def full_append(self, item):
        deque.append(self, item)
        # full, pop the oldest item, left most item
        self.popleft()

    def append(self, item):
        deque.append(self, item)
        # max size reached, append becomes full_append
        if len(self) == self.size:
            self.append = self.full_append
    def mean(self):
        l = self.get()
        return sum(l) / float(len(l))

    def get(self):
        """returns a list of size items (newest items)"""
        return list(self)

class RealModularMara3DOFv0Env(real_env.RealEnv):
    """
    This environment present a modular SCARA robot with a range finder at its
    end pointing towards the workspace of the robot. The goal of this environment is
    defined to reach the center of the "H" or the "O" from the "H-ROS" logo within the worspace.
    This environment uses `slowness=1` and matches the delay between actions/observations
    to this value (slowness). In other words, actions are taken at "1/slowness" rate.

    Reward is determined ... (TODO: describe the heuristic or reward calculation method)
    """
    def __init__(self):
        """
        Initialize the SCARA environemnt
            NOTE: This environment uses ROS and interfaces.

            TODO: port everything to ROS 2 natively
        """
        # Launch the simulation with the given launchfile name
        real_env.RealEnv.__init__(self)

        # TODO: cleanup this variables, remove the ones that aren't used
        # class variables
        self._observation_msg = None
        self.scale = None  # must be set from elsewhere based on observations
        self.bias = None
        self.x_idx = None
        self.obs = None
        self.reward = None
        self.done = None
        self.reward_dist = None
        self.reward_ctrl = None
        self.action_space = None
        self.max_episode_steps = 1000 # now used in all algorithms
        self.iterator = 0
        # default to seconds
        self.slowness = 1
        self.slowness_unit = 'sec'
        self.reset_jnts = True
        self.detect_target_once = 1

        self._time_lock = threading.RLock()

        #############################
        #   Environment hyperparams
        #############################
        # target, where should the agent reach
        EE_POS_TGT = np.asmatrix([-0.48731417, -0.00258965,  0.82467501])

        EE_ROT_TGT = np.asmatrix([[0.79660969, -0.51571238,  0.31536287], [0.51531424,  0.85207952,  0.09171542], [-0.31601302,  0.08944959,  0.94452874]])
        EE_POINTS = np.asmatrix([[0, 0, 0]])
        EE_VELOCITIES = np.asmatrix([[0, 0, 0]])
        # Initial joint position
        # INITIAL_JOINTS = np.array([0., 0., 0., 0., 0., 0.])
        INITIAL_JOINTS = np.array([0., 0., 0., 0., 0., 0.])
        # Used to initialize the robot, #TODO, clarify this more
        # STEP_COUNT = 2  # Typically 100.
        # slowness = 10000000 # 10 ms, where 1 second is real life simulation
        # slowness = 1000000 # 1 ms, where 1 second is real life simulation
        # slowness = 1 # use >10 for running trained network in the simulation
        # slowness = 10 # use >10 for running trained network in the simulation

        # Topics for the robot publisher and subscriber.
        JOINT_PUBLISHER = '/mara_controller/command'
        JOINT_SUBSCRIBER = '/mara_controller/state'

        # joint names:
        MOTOR1_JOINT = 'hros_actuation_servomotor_000A3500014E2'
        MOTOR2_JOINT = 'hros_actuation_servomotor_000A3500014E'
        MOTOR3_JOINT = 'hros_actuation_servomotor_000A3500018E'
        MOTOR4_JOINT = 'hros_actuation_servomotor_000A3500018E2'
        MOTOR5_JOINT = 'hros_actuation_servomotor_000A3500016E'
        MOTOR6_JOINT = 'hros_actuation_servomotor_000A3500016E2'

        # Set constants for links
        TABLE = 'table'

        BASE = 'base_link'

        MARA_MOTOR1_LINK = 'motor1_link'
        MARA_MOTOR2_LINK = 'motor2_link'
        MARA_MOTOR3_LINK = 'motor3_link'
        MARA_MOTOR4_LINK = 'motor4_link'
        MARA_MOTOR5_LINK = 'motor5_link'
        MARA_MOTOR6_LINK = 'motor6_link'
        EE_LINK = 'ee_link'


        # EE_LINK = 'ee_link'
        JOINT_ORDER = [MOTOR1_JOINT, MOTOR2_JOINT, MOTOR3_JOINT,
                       MOTOR4_JOINT, MOTOR5_JOINT, MOTOR6_JOINT]
        LINK_NAMES = [TABLE, BASE, MARA_MOTOR1_LINK, MARA_MOTOR2_LINK,
                            MARA_MOTOR3_LINK, MARA_MOTOR4_LINK,
                            MARA_MOTOR5_LINK, MARA_MOTOR6_LINK,
                      EE_LINK]

        reset_condition = {
            'initial_positions': INITIAL_JOINTS,
             'initial_velocities': []
        }
        #############################

        # TODO: fix this and make it relative
        # Set the path of the corresponding URDF file from "assets"
        URDF_PATH = '/home/erle/ros_catkin_ws/src/mara/mara_description/urdf/mara_demo_camera_top.urdf'

        m_joint_order = copy.deepcopy(JOINT_ORDER)
        m_link_names = copy.deepcopy(LINK_NAMES)
        m_joint_publishers = copy.deepcopy(JOINT_PUBLISHER)
        m_joint_subscribers = copy.deepcopy(JOINT_SUBSCRIBER)
        ee_pos_tgt = EE_POS_TGT
        ee_rot_tgt = EE_ROT_TGT

        # Initialize target end effector position
        ee_tgt = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T)
        self.realgoal = ee_tgt
        self.target_orientation = ee_rot_tgt

        self.environment = {
            # rk changed this to for the mlsh
            # 'ee_points_tgt': ee_tgt,
            'ee_points_tgt': self.realgoal,
            'ee_point_tgt_orient': self.target_orientation,
            'joint_order': m_joint_order,
            'link_names': m_link_names,
            # 'slowness': slowness,
            'reset_conditions': reset_condition,
            'tree_path': URDF_PATH,
            'joint_publisher': m_joint_publishers,
            'joint_subscriber': m_joint_subscribers,
            'end_effector_points': EE_POINTS,
            'end_effector_velocities': EE_VELOCITIES,
        }

        # Subscribe to the appropriate topics, taking into account the particular robot
        # ROS 1 implementation
        self._pub = rospy.Publisher(JOINT_PUBLISHER, JointTrajectory)
        self._sub = rospy.Subscriber(JOINT_SUBSCRIBER, JointTrajectoryControllerState, self.observation_callback)

        TARGET_SUBSCRIBER = '/mara/target'
        self._sub_tgt = rospy.Subscriber(TARGET_SUBSCRIBER, Pose, self.tgt_callback)

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
        # observation = self.take_observation()
        # assert not done
        # self.obs_dim = observation.size
        """
        obs_dim is defined as:
        num_dof + end_effector_points=3 + end_effector_velocities=3
        end_effector_points and end_effector_velocities is constant and equals 3
        recently also added quaternion to the obs, which has dimension=4
        """

        self.obs_dim = self.scara_chain.getNrOfJoints() + 10 #6 hardcode it for now

        # # Here idially we should find the control range of the robot. Unfortunatelly in ROS/KDL there is nothing like this.
        # # I have tested this with the mujoco enviroment and the output is always same low[-1.,-1.], high[1.,1.]
        # #bounds = self.model.actuator_ctrlrange.copy()
        low = -np.pi/2.0 * np.ones(self.scara_chain.getNrOfJoints())
        high = np.pi/2.0 * np.ones(self.scara_chain.getNrOfJoints())

        self.action_space = spaces.Box(low, high)
        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self.add_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        self.add_model_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.remove_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.pub_set_model = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

        # Seed the environment
        # Seed the environment
        self._seed()
        self.kf = []
        for i in range(6):
            # self.kf.append(KalmanFilter(initial_state_mean=0,
            #                             transition_matrices=[1],
            #                             observation_matrices=[1],
            #                             initial_state_covariance=1,
            #                             observation_covariance=5,
            #                             transition_covariance=0.001,
            #                             n_dim_obs=1, n_dim_state=1,
            #                             em_vars=['transition_covariance', 'observation_covariance', 'initial_state_covariance']))
            # self.kf.append(KalmanFilter(initial_state_mean=0,
            #                             transition_matrices=[1],
            #                             observation_matrices=[1],
            #                             initial_state_covariance=1,
            #                             observation_covariance=1,
            #                             transition_covariance=0.1,
            #                             n_dim_obs=1, n_dim_state=1))

            self.kf.append(RingBuffer(500))
    def tgt_callback(self,msg):
        # print("Whats the target?: ", msg)
        if self.detect_target_once is 1:
            print("Get the target from vision, for now just use position.")
            EE_POS_TGT = np.asmatrix([msg.position.x, msg.position.y, msg.position.z])
            EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            EE_POINTS = np.asmatrix([[0, 0, 0]])
            ee_pos_tgt = EE_POS_TGT

            # leave rotation target same since in scara we do not have rotation of the end-effector
            ee_rot_tgt = EE_ROT_TGT
            target1 = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T)

            # self.realgoal = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt_random1, ee_rot_tgt).T)

            self.realgoal = target1
            print("Predicted target is: ", self.realgoal)
            self.detect_target_once = 0

    def observation_callback(self, message):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        # index = 0
        # for value in message.actual.positions:
        #      message.actual.positions[index] = value * 3.14/180
        #      index = index + 1
        self._observation_msg =  message

    def init_time(self, slowness =1, slowness_unit='sec', reset_jnts=True):
            self.slowness = slowness
            self.slowness_unit = slowness_unit
            self.reset_jnts = reset_jnts
            print("slowness: ", self.slowness)
            print("slowness_unit: ", self.slowness_unit, "type of variable: ", type(slowness_unit))
            print("reset joints: ", self.reset_jnts, "type of variable: ", type(self.reset_jnts))

    def get_trajectory_message(self, action, robot_id=0):
        """
        Helper function.
        Wraps an action vector of joint angles into a JointTrajectory message.
        The velocities, accelerations, and effort do not control the arm motion
        """
        # Set up a trajectory message to publish.
        action_msg = JointTrajectory()
        action_msg.joint_names = self.environment['joint_order']
        # Create a point to tell the robot to move to.
        target = JointTrajectoryPoint()
        action_float = [float(i) for i in action]

        index = 0
        # print(self.kf)
        print(action_float)
        for value in action_float:
            # # self.kf[index] = self.kf[index].em(value, n_iter=5)
            # (filtered_state_means, filtered_state_covariances) = self.kf[index].filter(value)
            # (smoothed_state_means, smoothed_state_covariances) = self.kf[index].smooth(value)
            # action_float[index] = smoothed_state_means
            self.kf[index].append(value)
            action_float[index] = self.kf[index].mean()
            index = index + 1
        print(action_float)
        print('---------------------')

        target.positions = action_float
        target.velocities = [0]*6
        target.effort = [float('nan')]*6
        # These times determine the speed at which the robot moves:
        # it tries to reach the specified target position in 'slowness' time.
        if (self.slowness_unit == 'sec') or (self.slowness_unit is None):
            target.time_from_start.secs = self.slowness
        elif (self.slowness_unit == 'nsec'):
            target.time_from_start.nsecs = self.slowness
        else:
            print("Unrecognized unit. Please use sec or nsec.")

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

    def take_observation(self):
        """
        Take observation from the environment and return it.
        TODO: define return type
        """
        # Take an observation
        # done = False

        obs_message = self._observation_msg
        if obs_message is None:
            # print("last_observations is empty")
            return None

        # Collect the end effector points and velocities in
        # cartesian coordinates for the process_observationsstate.
        # Collect the present joint angles and velocities from ROS for the state.
        last_observations = self.process_observations(obs_message, self.environment)
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

            rotation_matrix = np.eye(4)
            rotation_matrix[:3, :3] = rot
            rotation_matrix[:3, 3] = trans

            # I need this calculations for the new reward function, need to send them back to the run mara or calculate them here
            current_quaternion = quaternion_from_matrix(rotation_matrix)
            tgt_quartenion = quaternion_from_matrix(self.target_orientation)

            A  = np.vstack([current_quaternion, np.ones(len(current_quaternion))]).T

            quat_error = current_quaternion - tgt_quartenion

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
                          np.reshape(quat_error, -1),
                          np.reshape(ee_velocities, -1),]

            return state

    def rmse_func(self, ee_points):
        """
        Computes the Residual Mean Square Error of the difference between current and desired end-effector position
        """
        rmse = np.sqrt(np.mean(np.square(ee_points), dtype=np.float32))
        return rmse

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Implement the environment step abstraction. Execute action and returns:
            - reward
            - done (status)
            - action
            - observation
            - dictionary (#TODO clarify)
        """
        self.iterator+=1

        self.reward_dist = -self.rmse_func(self.ob[self.scara_chain.getNrOfJoints():(self.scara_chain.getNrOfJoints()+3)])
        self.reward_orient = - self.rmse_func(self.ob[self.scara_chain.getNrOfJoints()+3:(self.scara_chain.getNrOfJoints()+7)])

        #scale here the orientation because it should not be the main bias of the reward, position should be
        orientation_scale = 0.01

        # here we want to fetch the positions of the end-effector which are nr_dof:nr_dof+3
        if(self.rmse_func(self.ob[self.scara_chain.getNrOfJoints():(self.scara_chain.getNrOfJoints()+3)])<0.005):
            self.reward = 1 - self.rmse_func(self.ob[self.scara_chain.getNrOfJoints():(self.scara_chain.getNrOfJoints()+3)]) # Make the reward increase as the distance decreases
            print("Reward is: ", self.reward)
        else:
            self.reward = self.reward_dist

        if(self.rmse_func(self.ob[self.scara_chain.getNrOfJoints()+3:(self.scara_chain.getNrOfJoints()+7)])<0.05):
            self.reward = self.reward +  orientation_scale * (1 -self.rmse_func(self.ob[self.scara_chain.getNrOfJoints()+3:(self.scara_chain.getNrOfJoints()+7)]))
            print("Reward orientation is: ", self.reward)
        else:
            self.reward = self.reward + orientation_scale * self.rmse_func(self.ob[self.scara_chain.getNrOfJoints()+3:(self.scara_chain.getNrOfJoints()+7)])

        # Calculate if the env has been solved
        done = bool(((abs(self.reward_dist) < 0.005) and (abs(self.reward_orient)) < 0.05) or (self.iterator>self.max_episode_steps))

        # Execute "action"
        self._pub.publish(self.get_trajectory_message(action[:self.scara_chain.getNrOfJoints()]))

        # # Take an observation
        # TODO: program this better, check that ob is not None, etc.
        self.ob = self.take_observation()
        while(self.ob is None):
            self.ob = self.take_observation()

        # Return the corresponding observations, rewards, etc.
        # TODO, understand better what's the last object to return

        return self.ob, self.reward, done, {}

    def publish_last_position(self, values):
        self._pub.publish(self.get_trajectory_message(values[:self.scara_chain.getNrOfJoints()]))

    def goToInit(self):
        self.ob = self.take_observation()
        while(self.ob is None):
            self.ob = self.take_observation()
        # # Go to initial position and wait until it arrives there
        # Wait until the arm is within epsilon of reset configuration.
        self._time_lock.acquire(True, -1)
        with self._time_lock:
            self._currently_resetting = True
        self._time_lock.release()

        if self._currently_resetting:
            epsilon = 1e-3
            reset_action = self.environment['reset_conditions']['initial_positions']
            now_action = self._observation_msg.actual.positions
            du = np.linalg.norm(reset_action-now_action, float(np.inf))
            self._pub.publish(self.get_trajectory_message(self.environment['reset_conditions']['initial_positions']))
            if du > epsilon:
                self._currently_resetting = True
                self._pub.publish(self.get_trajectory_message(self.environment['reset_conditions']['initial_positions']))
                time.sleep(3)

    def reset(self):
        """
        Reset the agent for a particular experiment condition.
        """

        self.iterator = 0

        if self.reset_jnts is True:
            self._pub.publish(self.get_trajectory_message(self.environment['reset_conditions']['initial_positions']))
            if (self.slowness_unit == 'sec') or (self.slowness_unit is None):
                time.sleep(int(self.slowness))
            elif(self.slowness_unit == 'nsec'):
                time.sleep(int(self.slowness/1000000000)) # using nanoseconds
            else:
                print("Unrecognized unit. Please use sec or nsec.")

        # Take an observation
        self.ob = self.take_observation()
        while(self.ob is None):
            self.ob = self.take_observation()

        # Return the corresponding observation
        return self.ob
