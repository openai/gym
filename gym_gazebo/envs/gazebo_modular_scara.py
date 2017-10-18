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

# ROS 2
import rclpy
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Used for publishing scara joint angles.
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import String


class GazeboModularScaraEnv(gazebo_env.GazeboEnv):
    """
    This environment present a modular SCARA robot with a range finder at its
    end pointing towards the workspace of the robot.
    """
    def __init__(self):
        """
        Initialize the SCARA environemnt
            NOTE: This environment uses ROS and ROS 2 interfaces,
                    for now communicating throught a bridge.

            TODO: port everything to ROS 2 natively
        """
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "ModularScara3_v0.launch")

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

        # Topics for the robot publisher and subscriber.
        JOINT_PUBLISHER = '/scara_controller/command'
        JOINT_SUBSCRIBER = '/scara_controller/state'
        # where should the agent reach
        EE_POS_TGT = np.asmatrix([0.3325683, 0.0657366, 0.3746])
        EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        EE_POINTS = np.asmatrix([[0, 0, 0]])
        EE_VELOCITIES = np.asmatrix([[0, 0, 0]])
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
        # Set end effector constants
        INITIAL_JOINTS = np.array([0, 0, 0])

        # TODO: modify the path
        # # where is your urdf? We load here the 3 joints.... In the agent_scara we need to generalize it for joints depending on the input urdf
        # TREE_PATH = '/home/rkojcev/catkin_ws/src/scara_e1/scara_e1_description/urdf/scara_e1_3joints.urdf'

        reset_condition = {
            'initial_positions': INITIAL_JOINTS,
             'initial_velocities': []
        }
        STEP_COUNT = 2  # Typically 100.
        # Set the number of seconds per step of a sample.
        TIMESTEP = 0.01  # Typically 0.01.
        # Set the number of timesteps per sample.
        STEP_COUNT = 100  # Typically 100.
        # Set the number of samples per condition.
        SAMPLE_COUNT = 5  # Typically 5.
        # set the number of conditions per iteration.
        # Set the number of trajectory iterations to collect.
        ITERATIONS = 20  # Typically 10.
        slowness = 2

        m_joint_order = copy.deepcopy(JOINT_ORDER)
        m_link_names = copy.deepcopy(LINK_NAMES)
        m_joint_publishers = copy.deepcopy(JOINT_PUBLISHER)
        m_joint_subscribers = copy.deepcopy(JOINT_SUBSCRIBER)
        ee_pos_tgt = EE_POS_TGT
        ee_rot_tgt = EE_ROT_TGT
        # Initialize target end effector position
        ee_tgt = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T)

        self.environment = {
            # 'type': AgentSCARAROS,
            'dt': TIMESTEP,
            'T': STEP_COUNT,
            'ee_points_tgt': ee_tgt,
            'joint_order': m_joint_order,
            'link_names': m_link_names,
            'slowness': slowness,
            'reset_conditions': reset_condition,
            'tree_path': TREE_PATH,
            'joint_publisher': m_joint_publishers,
            'joint_subscriber': m_joint_subscribers,
            'end_effector_points': EE_POINTS,
            'end_effector_velocities': EE_VELOCITIES,
            'num_samples': SAMPLE_COUNT,
        }

        # TODO: review
        # self.spec = {'timestep_limit': 5, 'reward_threshold':  950.0,}

        # Subscribe to the appropriate topics, taking into account the particular robot
        ## ROS 1 implementation
        #self._pub = rospy.Publisher('/scara_controller/command', JointTrajectory)
        #self._sub = rospy.Subscriber('/scara_controller/state', JointTrajectoryControllerState, self._observation_callback)

        # ROS 2 implementation, includes initialization of the appropriate ROS 2 abstractions
        rclpy.init(args=None)
        self.ros2_node = rclpy.create_node('robot_ai_node')
        self._pub = ros2_node.create_publisher(JointTrajectory,'/scara_controller/command')
        # self._callbacks = partial(self._observation_callback, robot_id=0)
        self._sub = ros2_node.create_subscription(JointTrajectoryControllerState, '/scara_controller/state', self.observation_callback, qos_profile=qos_profile_sensor_data)
        self._time_lock = threading.RLock()

        #TODO set up the path appropriately

        # Initialize a tree structure from the robot urdf.
        #   note that the xacro of the urdf is updated by hand.
        # The urdf must be compiled.

        # TODO review with Risto
        _, self.ur_tree = treeFromFile(self.agent['tree_path'])
        # Retrieve a chain structure between the base and the start of the end effector.
        self.ur_chain = self.ur_tree.getChain(self.agent['link_names'][0], self.agent['link_names'][-1])
        print("nr of jnts: ", self.ur_chain.getNrOfJoints())
        # Initialize a KDL Jacobian solver from the chain.
        self.jac_solver = ChainJntToJacSolver(self.ur_chain)
        #print(self.jac_solver)
        self._observations_stale = [False for _ in range(1)]
        #print("after observations stale")
        self._currently_resetting = [False for _ in range(1)]
        self.reset_joint_angles = [None for _ in range(1)]

        # TODO review with Risto
        # # taken from mujoco in OpenAi how to initialize observation space and action space.
        # observation, _reward, done, _info = self._step(np.zeros(self.ur_chain.getNrOfJoints()))
        # assert not done
        # self.obs_dim = observation.size
        # # print(observation, _reward)
        # # Here idially we should find the control range of the robot. Unfortunatelly in ROS/KDL there is nothing like this.
        # # I have tested this with the mujoco enviroment and the output is always same low[-1.,-1.], high[1.,1.]
        # #bounds = self.model.actuator_ctrlrange.copy()
        # low = -np.pi/2.0 * np.ones(self.ur_chain.getNrOfJoints())#bounds[:, 0]
        # high = np.pi/2.0 * np.ones(self.ur_chain.getNrOfJoints()) #bounds[:, 1]
        # print("Action spaces: ", low, high)
        # self.action_space = spaces.Box(low, high)
        # high = np.inf*np.ones(self.obs_dim)
        # low = -high
        # self.observation_space = spaces.Box(low, high)

        self.action_space = spaces.Discrete(3) #F,L,R
        self.reward_range = (-np.inf, np.inf)

        # Gazebo specific services to start/stop its behavior and
        # facilitate the overall RL environment
        self.unpause = roJointTrajectoryControllerStatespy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # Seed the environment
        self._seed()

    def observation_callback(self, message):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        self._observation_msg =  message

    # def discretize_observation(self,data,new_ranges):
    #     discretized_ranges = []
    #     min_range = 0.2
    #     done = False
    #     mod = len(data.ranges)/new_ranges
    #     for i, item in enumerate(data.ranges):
    #         if (i%mod==0):
    #             if data.ranges[i] == float ('Inf') or np.isinf(data.ranges[i]):
    #                 discretized_ranges.append(6)
    #             elif np.isnan(data.ranges[i]):
    #                 discretized_ranges.append(0)
    #             else:
    #                 discretized_ranges.append(int(data.ranges[i]))
    #         if (min_range > data.ranges[i] > 0):
    #             done = True
    #     return discretized_ranges,done

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
        target.time_from_start.sec = self.environment['slowness']
        # Package the single point into a trajectory of points with length 1.
        action_msg.points = [target]
        return action_msg

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        """
        Implement the environment step abstraction. Execute action and returns:
            - observation
            - reward
            - done (status)
            - dictionary (#TODO clarify)
        """
        # Unpause simulation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # Execute "action"
        if rclpy.ok():
            self._pub.publish(self.get_trajectory_message(action[:3]))


        # Pause simulation
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        # Obtain observation

        # Calculate reward based on observation

        # Return the corresponding observations, rewards, etc.
        return state, reward, done, {}

    def _reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state = self.discretize_observation(data,5)

        return state
