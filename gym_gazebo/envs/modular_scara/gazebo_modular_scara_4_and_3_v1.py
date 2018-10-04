import gym
import rospy
import roslaunch
import time
import numpy as np
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from gym.utils import seeding
import copy

import threading # Used for time locks to synchronize position data.

# ROS 2
# import rclpy
# from rclpy.qos import QoSProfile, qos_profile_sensor_data
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Used for publishing scara joint angles.
# from control_msgs.msg import JointTrajectoryControllerState
# from std_msgs.msg import String

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


class GazeboModularScara4And3DOFv1Env(gazebo_env.GazeboEnv):
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
        gazebo_env.GazeboEnv.__init__(self, "ModularScara4_3_v0.launch")

        # TODO: cleanup this variables, remove the ones that aren't used
        # class variables

        self.urdf_path = "/home/rkojcev/devel/ros_rl/environments/gym-gazebo/gym_gazebo/envs/assets/urdf/modular_scara/scara_e1_model_4_and_3joints.urdf"

        self.slowness = 1
        self.slowness_unit = 'sec'

        #self.slowness = 10000000
        #self.slowness_unit = 'nsec'

        #choose random enviroment at the startup
        self.randomizeRobot()
        # Seed the environment
        self._seed()

    def init_3dof_robot(self):
        print("I am in enviroment 3DOF")
        self._observation_msg = None
        print(self._observation_msg)
        self.scale = None  # must be set from elsewhere based on observations
        self.bias = None
        self.x_idx = None
        self.obs = None
        self.reward = None
        self.done = None
        self.reward_dist = None
        self.action_space = None
        self.max_episode_steps = 1000 # now used in all algorithms
        self.iterator = 0

        self._time_lock = threading.RLock()
        self.choose_robot = 0
        self.environment = None

        #############################
        #   Environment hyperparams
        #############################
        # target, where should the agent reach
        # EE_POS_TGT = np.asmatrix([0.3325683, 0.0657366, 0.3746]) # center of O
        EE_POS_TGT = np.asmatrix([0.3305805, -0.1326121, 0.3746]) # center of the H
        EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        EE_POINTS = np.asmatrix([[0, 0, 0]])
        EE_VELOCITIES = np.asmatrix([[0, 0, 0]])
        # Initial joint position
        INITIAL_JOINTS = np.array([0., 0., 0.])
        # Used to initialize the robot, #TODO, clarify this more
        STEP_COUNT = 2  # Typically 100.

        # joint names:
        MOTOR1_JOINT = 'motor1'
        MOTOR2_JOINT = 'motor2'
        MOTOR3_JOINT = 'motor3'
        # Set constants for links
        WORLD = "world"
        BASE = 'scara_e1_base_link'
        BASE_MOTOR = 'scara_e1_base_motor'
        #
        SCARA_MOTOR1 = 'scara_e1_motor1'
        SCARA_INSIDE_MOTOR1 = 'scara_e1_motor1_inside'
        SCARA_SUPPORT_MOTOR1 = 'scara_e1_motor1_support'
        SCARA_BAR_MOTOR1 = 'scara_e1_bar1'
        SCARA_FIXBAR_MOTOR1 = 'scara_e1_fixbar1'
        #
        SCARA_MOTOR2 = 'scara_e1_motor2'
        SCARA_INSIDE_MOTOR2 = 'scara_e1_motor2_inside'
        SCARA_SUPPORT_MOTOR2 = 'scara_e1_motor2_support'
        SCARA_BAR_MOTOR2 = 'scara_e1_bar2'
        SCARA_FIXBAR_MOTOR2 = 'scara_e1_fixbar2'
        #
        SCARA_MOTOR3 = 'scara_e1_motor3'
        SCARA_INSIDE_MOTOR3 = 'scara_e1_motor3_inside'
        SCARA_SUPPORT_MOTOR3 = 'scara_e1_motor3_support'
        SCARA_BAR_MOTOR3 = 'scara_e1_bar3'
        SCARA_FIXBAR_MOTOR3 = 'scara_e1_fixbar3'
        #
        SCARA_RANGEFINDER = 'scara_e1_rangefinder'
        EE_LINK = 'ee_link'
        JOINT_ORDER = [MOTOR1_JOINT, MOTOR2_JOINT, MOTOR3_JOINT]
        LINK_NAMES = [BASE, BASE_MOTOR,
                      SCARA_MOTOR1, SCARA_INSIDE_MOTOR1, SCARA_SUPPORT_MOTOR1, SCARA_BAR_MOTOR1, SCARA_FIXBAR_MOTOR1,
                      SCARA_MOTOR2, SCARA_INSIDE_MOTOR2, SCARA_SUPPORT_MOTOR2, SCARA_BAR_MOTOR2, SCARA_FIXBAR_MOTOR2,
                      SCARA_MOTOR3, SCARA_INSIDE_MOTOR3, SCARA_SUPPORT_MOTOR3,
                      EE_LINK]

        reset_condition = {
            'initial_positions': INITIAL_JOINTS,
             'initial_velocities': []
        }
        #############################

        # TODO: fix this and make it relative
        # Set the path of the corresponding URDF file from "assets"
        m_joint_order = copy.deepcopy(JOINT_ORDER)
        m_link_names = copy.deepcopy(LINK_NAMES)
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
            'reset_conditions': reset_condition,
            'tree_path': self.urdf_path,
            'end_effector_points': EE_POINTS,
            'end_effector_velocities': EE_VELOCITIES,
        }
        # self.spec = {'timestep_limit': 5, 'reward_threshold':  950.0,}

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

        # Topics for the robot publisher and subscriber.
        # self.enviroment = None
        JOINT_PUBLISHER = '/scara_controller_3dof/command'
        JOINT_SUBSCRIBER = '/scara_controller_3dof/state'

        # Subscribe to the appropriate topics, taking into account the particular robot
        # ROS 1 implementation
        self._pub_3dof = rospy.Publisher(JOINT_PUBLISHER, JointTrajectory)
        self._sub_3dof = rospy.Subscriber(JOINT_SUBSCRIBER, JointTrajectoryControllerState, self.observation_callback)


    def init_4dof_robot(self):
        print("I am in enviroment 4DOF")
        self._observation_msg_4dof = None
        print(self._observation_msg_4dof)
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
        self.reset_jnts = False
        self._time_lock = threading.RLock()
        self.choose_robot = 1
        self.environment = None

        #############################
        #   Environment hyperparams
        #############################

        ########################################################################################
        '''
        Add here the stuff needed for the 4DOF robot
        '''

        EE_POS_TGT_4DOF = np.asmatrix([0.3305805, -0.1326121, 0.4868]) # center of the H
        EE_ROT_TGT_4DOF = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        EE_POINTS_4DOF = np.asmatrix([[0, 0, 0]])
        EE_VELOCITIES_4DOF = np.asmatrix([[0, 0, 0]])
        # Initial joint position
        INITIAL_JOINTS_4DOF = np.array([0., 0., 0., 0.])

        # JOINT_PUBLISHER_4DOF = '/scara_controller_4dof/command'
        # JOINT_SUBSCRIBER_4DOF = '/scara_controller_4dof/state'
        # joint names:
        MOTOR1_4DOF_JOINT = 'motor1_4dof'
        MOTOR2_4DOF_JOINT = 'motor2_4dof'
        MOTOR3_4DOF_JOINT = 'motor3_4dof'
        MOTOR4_4DOF_JOINT = 'motor4_4dof'

        # Set constants for links
        BASE_4DOF = '4_dof_scara_e1_base_link'
        BASE_4DOF_MOTOR = '4_dof_scara_e1_base_motor'

        SCARA_4DOF_MOTOR1 = '4_dof_scara_e1_motor1'
        SCARA_4DOF_INSIDE_MOTOR1 = '4_dof_scara_e1_motor1_inside'
        SCARA_4DOF_SUPPORT_MOTOR1 = '4_dof_scara_e1_motor1_support'
        SCARA_4DOF_BAR_MOTOR1 = '4_dof_scara_e1_bar1'
        SCARA_4DOF_FIXBAR_MOTOR1 = '4_dof_scara_e1_fixbar1'

        SCARA_4DOF_MOTOR2 = '4_dof_scara_e1_motor2'
        SCARA_4DOF_INSIDE_MOTOR2 = '4_dof_scara_e1_motor2_inside'
        SCARA_4DOF_SUPPORT_MOTOR2 = '4_dof_scara_e1_motor2_support'
        SCARA_4DOF_BAR_MOTOR2 = '4_dof_scara_e1_bar2'
        SCARA_4DOF_FIXBAR_MOTOR2 = '4_dof_scara_e1_fixbar2'

        SCARA_4DOF_MOTOR3 = '4_dof_scara_e1_motor3'
        SCARA_4DOF_INSIDE_MOTOR3 = '4_dof_scara_e1_motor3_inside'
        SCARA_4DOF_SUPPORT_MOTOR3 = '4_dof_scara_e1_motor3_support'
        SCARA_4DOF_BAR_MOTOR3 = '4_dof_scara_e1_bar3'
        SCARA_4DOF_FIXBAR_MOTOR3 = '4_dof_scara_e1_fixbar3'

        SCARA_4DOF_MOTOR4 = '4_dof_scara_e1_motor4'
        SCARA_4DOF_INSIDE_MOTOR4 = '4_dof_scara_e1_motor4_inside'
        SCARA_4DOF_SUPPORT_MOTOR4 = '4_dof_scara_e1_motor4_support'
        SCARA_4DOF_BAR_MOTOR4 = '4_dof_scara_e1_bar4'
        SCARA_4DOF_FIXBAR_MOTOR4= '4_dof_scara_e1_fixbar4'

        SCARA_4DOF_RANGEFINDER = '4_dof_scara_e1_rangefinder'

        EE_LINK_4DOF= '4_dof_ee_link'
        JOINT_ORDER_4DOF = [MOTOR1_4DOF_JOINT, MOTOR2_4DOF_JOINT, MOTOR3_4DOF_JOINT, MOTOR4_4DOF_JOINT]
        LINK_NAMES_4DOF = [BASE_4DOF, BASE_4DOF_MOTOR,
              SCARA_4DOF_MOTOR1, SCARA_4DOF_INSIDE_MOTOR1, SCARA_4DOF_SUPPORT_MOTOR1, SCARA_4DOF_BAR_MOTOR1, SCARA_4DOF_FIXBAR_MOTOR1,
              SCARA_4DOF_MOTOR2, SCARA_4DOF_INSIDE_MOTOR2, SCARA_4DOF_SUPPORT_MOTOR2, SCARA_4DOF_BAR_MOTOR2, SCARA_4DOF_FIXBAR_MOTOR2,
              SCARA_4DOF_MOTOR3, SCARA_4DOF_INSIDE_MOTOR3, SCARA_4DOF_SUPPORT_MOTOR3, SCARA_4DOF_BAR_MOTOR3, SCARA_4DOF_FIXBAR_MOTOR3,
              SCARA_4DOF_MOTOR4, SCARA_4DOF_INSIDE_MOTOR4, SCARA_4DOF_SUPPORT_MOTOR4,
              EE_LINK_4DOF]

        reset_condition = {
            'initial_positions': INITIAL_JOINTS_4DOF,
             'initial_velocities': []
        }
        #################################################################################################################################################

        # TODO: fix this and make it relative
        # Set the path of the corresponding URDF file from "assets"


        m_joint_order = copy.deepcopy(JOINT_ORDER_4DOF)
        m_link_names = copy.deepcopy(LINK_NAMES_4DOF)
        ee_pos_tgt = EE_POS_TGT_4DOF
        ee_rot_tgt = EE_ROT_TGT_4DOF

        # Initialize target end effector position
        ee_tgt = np.ndarray.flatten(get_ee_points(EE_POINTS_4DOF, ee_pos_tgt, ee_rot_tgt).T)
        self.realgoal = ee_tgt

        self.environment = {
            'ee_points_tgt': self.realgoal,
            'joint_order': m_joint_order,
            'link_names': m_link_names,
            'reset_conditions': reset_condition,
            'tree_path': self.urdf_path,
            # 'joint_publisher': m_joint_publishers,
            # 'joint_subscriber': m_joint_subscribers,
            'end_effector_points': EE_POINTS_4DOF,
            'end_effector_velocities': EE_VELOCITIES_4DOF,
        }

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
        """
        #
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

        # Topics for the robot publisher and subscriber.
        # self.enviroment = None
        JOINT_PUBLISHER_4DOF = '/scara_controller_4dof/command'
        JOINT_SUBSCRIBER_4DOF = '/scara_controller_4dof/state'

        # Subscribe to the appropriate topics, taking into account the particular robot
        # ROS 1 implementation
        self._pub_4dof = rospy.Publisher('/scara_controller_4dof/command', JointTrajectory)
        self._sub_4dof = rospy.Subscriber('/scara_controller_4dof/state', JointTrajectoryControllerState, self.observation_callback_4dof)

    def observation_callback(self, message):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        self._observation_msg =  message
    def observation_callback_4dof(self, message):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        self._observation_msg_4dof =  message
    #def init_time(self, slowness =1, slowness_unit='sec', reset_jnts=True):
    def init_time(self, slowness = 1, slowness_unit='sec', reset_jnts=False):
            self.slowness = slowness
            self.slowness_unit = slowness_unit
            self.reset_jnts = reset_jnts
            print("slowness: ", self.slowness)
            print("slowness_unit: ", self.slowness_unit, "type of variable: ", type(slowness_unit))
            print("reset joints: ", self.reset_jnts, "type of variable: ", type(self.reset_jnts))

    def randomizeTarget(self):
        print("calling randomize target")
        if self.choose_robot is 0:
            print("Randomize target for the 3 DoF")
            EE_POS_TGT_1 = np.asmatrix([0.3325683, 0.0657366, 0.3746]) # center of O
            EE_POS_TGT_2 = np.asmatrix([0.3305805, -0.1326121, 0.3746]) # center of the H
            EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            EE_POINTS = np.asmatrix([[0, 0, 0]])

            ee_pos_tgt_1 = EE_POS_TGT_1
            ee_pos_tgt_2 = EE_POS_TGT_2

            # leave rotation target same since in scara we do not have rotation of the end-effector
            ee_rot_tgt = EE_ROT_TGT

            # Initialize target end effector position
            target1 = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt_1, ee_rot_tgt).T)
            target2 = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt_2, ee_rot_tgt).T)

            """
            This is for initial test only, we need to change this in the future to be more realistic.
            E.g. covered target -> go to other target. This could be implemented for example with vision.
            """
            self.realgoal = target1 if np.random.uniform() < 0.5 else target2
            print("randomizeTarget realgoal: ", self.realgoal)
        else:
            print("Randomize target for the 4 DoF")
            EE_POS_TGT_1 = np.asmatrix([0.3325683, 0.0657366, 0.4868]) # center of O
            EE_POS_TGT_2 = np.asmatrix([0.3305805, -0.1326121, 0.4868]) # center of the H
            EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            EE_POINTS = np.asmatrix([[0, 0, 0]])

            ee_pos_tgt_1 = EE_POS_TGT_1
            ee_pos_tgt_2 = EE_POS_TGT_2

            # leave rotation target same since in scara we do not have rotation of the end-effector
            ee_rot_tgt = EE_ROT_TGT

            target1 = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt_1, ee_rot_tgt).T)
            target2 = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt_2, ee_rot_tgt).T)

            """
            This is for initial test only, we need to change this in the future to be more realistic.
            E.g. covered target -> go to other target. This could be implemented for example with vision.
            """
            self.realgoal = target1 if np.random.uniform() < 0.5 else target2
            print("randomizeTarget realgoal: ", self.realgoal)


    def randomizeRobot(self):
        print("calling randomize correct")
        #try to choose environment
        if np.random.uniform() < 0.5:
            self.choose_robot=0
            print("which robot is? ", self.choose_robot)
            enviroment_test = self.init_3dof_robot()
        else:
            self.choose_robot=1
            print("which robot is?", self.choose_robot)
            enviroment_test = self.init_4dof_robot()

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
        target.positions = action_float
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
            # print('joint messages: ', message.joint_names)
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
        if self.choose_robot is 0:
            obs_message = self._observation_msg
        else:
            obs_message = self._observation_msg_4dof
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
            if self.choose_robot is 0:
                last_observations = np.insert(last_observations, 3, 0.)
                # print('last_observations_extension: ', last_observations)
                # last_observations.append(0.0)
            return np.r_[np.reshape(last_observations, -1),
                          np.reshape(ee_points, -1),
                          np.reshape(ee_velocities, -1),]

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

        # here we want to fetch the positions of the end-effector which are nr_dof:nr_dof+3
        if(self.rmse_func(self.ob[self.scara_chain.getNrOfJoints():(self.scara_chain.getNrOfJoints()+3)])<0.005):
            self.reward = 1 - self.rmse_func(self.ob[self.scara_chain.getNrOfJoints():(self.scara_chain.getNrOfJoints()+3)]) # Make the reward increase as the distance decreases
            print("Reward is: ", self.reward)
        else:
            self.reward = self.reward_dist

        # Calculate if the env has been solved
        done = bool(abs(self.reward_dist) < 0.005) or (self.iterator>self.max_episode_steps)

        # Execute "action"
        if self.choose_robot is 0:
            self._pub_3dof.publish(self.get_trajectory_message(action[:self.scara_chain.getNrOfJoints()]))
        else:
            self._pub_4dof.publish(self.get_trajectory_message(action[:self.scara_chain.getNrOfJoints()]))

        # # Take an observation
        # TODO: program this better, check that ob is not None, etc.
        self.ob = self.take_observation()
        while(self.ob is None):
            self.ob = self.take_observation()

        # Return the corresponding observations, rewards, etc.
        ee_point = self.ob[self.scara_chain.getNrOfJoints():(self.scara_chain.getNrOfJoints()+3)] + self.realgoal
        ee_point_eucledian = np.linalg.norm(self.ob[self.scara_chain.getNrOfJoints():(self.scara_chain.getNrOfJoints()+3)])
        return self.ob, self.reward, done, {}, ee_point, ee_point_eucledian

    def reset(self):
        """
        Reset the agent for a particular experiment condition.
        """
        self.iterator = 0

        if self.reset_jnts is True:
            if self.choose_robot is 0:
                self._pub_3dof.publish(self.get_trajectory_message(self.environment['reset_conditions']['initial_positions']))
            elif self.choose_robot is 1:
                self._pub_4dof.publish(self.get_trajectory_message(self.environment['reset_conditions']['initial_positions']))

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
