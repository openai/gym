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
import rospkg
import random

from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import Image
from baselines.agent.scara_arm.tree_urdf import treeFromFile # For KDL Jacobians
from PyKDL import Jacobian, Chain, ChainJntToJacSolver, JntArray # For KDL Jacobians


# from custom baselines repository
from baselines.agent.utility.general_utils import forward_kinematics, get_ee_points, rotation_from_matrix, \
    get_rotation_matrix,quaternion_from_matrix # For getting points and velocities.

class MSG_INVALID_JOINT_NAMES_DIFFER(Exception):
    """Error object exclusively raised by _process_observations."""
    pass

class ROBOT_MADE_CONTACT_WITH_GAZEBO_GROUND_SO_RESTART_ROSLAUNCH(Exception):
    """Error object exclusively raised by reset."""
    pass

class Box3DOFv1Env(gazebo_env.GazeboEnv):

    def __init__(self):

        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "Box_Vision_v0.launch")

        # TODO: cleanup this variables, remove the ones that aren't used
        # class variables
        self._observation_msg = None
        self._camera_image = None
        self.scale = None  # must be set from elsewhere based on observations
        self.bias = None
        self.x_idx = None
        self.obs = None
        self.reward = None
        self.done = None
        self.reward_dist = None
        self.reward_ctrl = None
        self.action_space = None

        #############################
        #   Environment hyperparams
        #############################
        # target, where should the agent reach
        EE_POS_TGT = np.asmatrix([0.3325683, 0.0657366, 0.3746])
        EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        EE_POINTS = np.asmatrix([[0, 0, 0]])
        EE_VELOCITIES = np.asmatrix([[0, 0, 0]])
        # Initial joint position
        INITIAL_JOINTS = np.array([0, 0, 0])
        # Used to initialize the robot, #TODO, clarify this more
        STEP_COUNT = 2  # Typically 100.
        # # Set the number of seconds per step of a sample.
        # TIMESTEP = 0.01  # Typically 0.01.
        # # Set the number of timesteps per sample.
        # STEP_COUNT = 100  # Typically 100.
        # # Set the number of samples per condition.
        # SAMPLE_COUNT = 5  # Typically 5.
        # # Set the number of trajectory iterations to collect.
        # ITERATIONS = 20  # Typically 10.
        # How much time does it take to execute the trajectory (in seconds)
        slowness = 2

        # Topics for the robot publisher and subscriber.
        JOINT_PUBLISHER = '/scara_controller/command'
        JOINT_SUBSCRIBER = '/scara_controller/state'
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

        #############################
        rospack = rospkg.RosPack()
        prefix_path = rospack.get_path('scara_e1_description')

        if(prefix_path==None):
            print("I can't find scara_e1_description")
            sys.exit(0)

        URDF_PATH = prefix_path + "/urdf/scara_e1_model_3joints_simplified_camera_behind.urdf.xacro"


        m_joint_order = copy.deepcopy(JOINT_ORDER)
        m_link_names = copy.deepcopy(LINK_NAMES)
        m_joint_publishers = copy.deepcopy(JOINT_PUBLISHER)
        m_joint_subscribers = copy.deepcopy(JOINT_SUBSCRIBER)
        ee_pos_tgt = EE_POS_TGT
        ee_rot_tgt = EE_ROT_TGT
        # Initialize target end effector position
        ee_tgt = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T)

        self.environment = {
            # 'dt': TIMESTEP,
            'T': STEP_COUNT,
            'ee_points_tgt': ee_tgt,
            'joint_order': m_joint_order,
            'link_names': m_link_names,
            'slowness': slowness,
            'reset_conditions': reset_condition,
            'tree_path': URDF_PATH,
            'joint_publisher': m_joint_publishers,
            'joint_subscriber': m_joint_subscribers,
            'end_effector_points': EE_POINTS,
            'end_effector_velocities': EE_VELOCITIES,
        }

        # self.spec = {'timestep_limit': 5, 'reward_threshold':  950.0,}

        # Subscribe to the appropriate topics, taking into account the particular robot
        # ROS 1 implementation
        self._pub = rospy.Publisher('/scara_controller/command', JointTrajectory)
        self._sub = rospy.Subscriber('/scara_controller/state', JointTrajectoryControllerState, self.observation_callback)

        self._image_sub = rospy.Subscriber('/scara/camera/image_raw', Image, self.camera_callback)

        #self._sub = rospy.Subscriber('/scara/camera', JointTrajectoryControllerState, self.observation_callback)

        # # ROS 2 implementation, includes initialization of the appropriate ROS 2 abstractions
        # rclpy.init(args=None)
        # self.ros2_node = rclpy.create_node('robot_ai_node')
        # self._pub = ros2_node.create_publisher(JointTrajectory, JOINT_PUBLISHER)
        # # self._callbacks = partial(self.observation_callback, robot_id=0)
        # self._sub = ros2_node.create_subscription(JointTrajectoryControllerState, JOINT_SUBSCRIBER, self.observation_callback, qos_profile=qos_profile_sensor_data)
        # # self._time_lock = threading.RLock()

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
        self.obs_dim = 9 # hardcode it for now
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
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.add_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        # Seed the environment
        self._seed()


        model_xml = "<?xml version=\"1.0\"?> \
                    <robot name=\"myfirst\"> \
                      <link name=\"world\"> \
                      </link>\
                      <link name=\"cylinder0\">\
                        <visual>\
                          <geometry>\
                            <sphere radius=\"0.01\"/>\
                          </geometry>\
                          <origin xyz=\"0 0 0\"/>\
                          <material name=\"rojotransparente\">\
                              <ambient>0.5 0.5 1.0 0.1</ambient>\
                              <diffuse>0.5 0.5 1.0 0.1</diffuse>\
                          </material>\
                        </visual>\
                        <inertial>\
                          <mass value=\"5.0\"/>\
                          <inertia ixx=\"1.0\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"1.0\" iyz=\"0.0\" izz=\"1.0\"/>\
                        </inertial>\
                      </link>\
                      <joint name=\"world_to_base\" type=\"fixed\"> \
                        <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\
                        <parent link=\"world\"/>\
                        <child link=\"cylinder0\"/>\
                      </joint>\
                      <gazebo reference=\"cylinder0\">\
                        <material>Gazebo/GreenTransparent</material>\
                      </gazebo>\
                    </robot>"
        robot_namespace = ""
        pose = Pose()
        pose.position.x = EE_POS_TGT[0,0];
        pose.position.y = EE_POS_TGT[0,1];
        pose.position.z = 0.055#EE_POS_TGT[0,2];

        #Static obstacle (not in original code)
        # pose.position.x = 0.25;#
        # pose.position.y = 0.07;#
        # pose.position.z = 0.0;#

        pose.orientation.x = 0;
        pose.orientation.y= 0;
        pose.orientation.z = 0;
        pose.orientation.w = 0;
        reference_frame = ""
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        self.add_model(model_name="target",
                        model_xml=model_xml,
                        robot_namespace="",
                        initial_pose=pose,
                        reference_frame="")

    def observation_callback(self, message):
        """
        Callback method for the subscriber of JointTrajectoryControllerState
        """
        self._observation_msg =  message
        # print("!!!!!!!!!Received an observation")


    def camera_callback(self, message):
        """
        Callback method for the subscriber of Image
        """
        self._camera_image =  message
        print("\ncamera_image.height",self._camera_image.height)
        print("\ncamera_image.width",self._camera_image.width)

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
        target.time_from_start.secs = self.environment['slowness']
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
        # Initialize a Jacobian for 6 joint angles by 3 cartesian coords and 3 orientation angles
        jacobian = Jacobian(3)
        # Initialize a joint array for the present 6 joint angles.
        angles = JntArray(3)
        # Construct the joint array from the most recent joint angles.
        for i in range(3):
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
                                            (0, 2, 1)).reshape(-1, 3)
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

    def addObstacle(self):
        print("\nADD OBSTACLE")
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        try:
            # #NORMAL BOX
            model_xml = "<?xml version=\"1.0\"?> \
                                <robot name=\"myfirst\"> \
                                <link name=\"world\"> \
                                </link>\
                                <link name=\"cylinder0\">\
                                <visual>\
                                    <geometry>\
                                        <box size=\".05 .05 .05\"/>\
                                        </geometry>\
                                    <origin xyz=\"0 0 0\"/>\
                                    <material name=\"red\">\
                                        <ambient>0.5 0.5 1.0 0.1</ambient>\
                                        <diffuse>0.5 0.5 1.0 0.1</diffuse>\
                                    </material>\
                                </visual>\
                                <inertial>\
                                    <mass value=\"10.\"/>\
                                    <inertia ixx=\"1\" ixy=\".0\" ixz=\"0.0\" iyy=\"1\" iyz=\"0.0\" izz=\"1\"/>\
                                </inertial>\
                                <collision>\
                                    <geometry>\
                                        <box size=\".05 .05 .05\"/>\
                                    </geometry>\
                                </collision>\
                                </link>\
                                <joint name=\"world_to_base\" type=\"fixed\"> \
                                    <origin xyz=\"0 0 0.3\" rpy=\"0 0 0\"/>\
                                    <parent link=\"world\"/>\
                                    <child link=\"cylinder0\"/>\
                                </joint>\
                                <gazebo reference=\"cylinder0\">\
                                    <material>Gazebo/Red</material>\
                                </gazebo>\
                        </robot>"
            robot_namespace = ""
            pose = Pose()
            number = random.randint(0, 4)
            #pose.position.x = 0.15;#
            pose.position.x = 0.3;#
            pose.position.y = 0.07;#
            pose.position.z = 0.05;
            pose.orientation.x = 0;
            pose.orientation.y= 0;
            pose.orientation.z = 0;
            pose.orientation.w = 0;

            reference_frame = ""

            self.add_model(model_name="obstacle",
                                        model_xml=model_xml,
                                        robot_namespace=robot_namespace,
                                        initial_pose=pose,
                                        reference_frame="")
        except (rospy.ServiceException) as e:
                print ("/gazebo/spawn_urdf_model service call failed")


    def take_observation(self):
        """
        Take observation from the environment and return it.
        TODO: define return type
        """
        # Take an observation
        # done = False

        # # ROS 2, Acquire the lock to prevent the subscriber thread from
        # # updating times or observation messages.
        # self._time_lock.acquire(True)
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
                                        last_observations[:3],
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
            ee_points = current_ee_tgt - self.environment['ee_points_tgt']
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
        # if rclpy.ok(): # ROS 2 code
        self._pub.publish(self.get_trajectory_message(action[:3]))

        # # Take an observation
        # TODO: program this better, check that ob is not None, etc.
        self.ob = self.take_observation()
        while(self.ob is None):
            self.ob = self.take_observation()

        # Pause simulation
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        # Take an observation
        # TODO: program this better, check that ob is not None, etc.
        # self.ob = take_observation()

        self.reward_dist = - self.rmse_func(self.ob[2])
        if abs(self.reward_dist) < 0.005:
            # print("reward (Eucledian dist (mm)): ", -1000 * self.reward_dist)
            self.reward = 100 + self.reward_dist
        else:
            # print("reward (Eucledian dist (mm)): ", -1000 * self.reward_dist)
            self.reward = self.reward_dist

        # print("reward: ", self.reward)

        # Calculate if the env has been solved
        done = bool(abs(self.reward_dist) < 0.005)

        # self._time_lock.release() # ROS 2 code

        # Return the corresponding observations, rewards, etc.
        # TODO, understand better what's the last object to return
        return self.ob, self.reward, done, {}

    def reset(self):
        """
        Reset the agent for a particular experiment condition.
        """
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # Take an observation
        self.ob = self.take_observation()
        while(self.ob is None):
            self.ob = self.take_observation()

        # Pause the simulation
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        # Return the corresponding observation
        return self.ob
