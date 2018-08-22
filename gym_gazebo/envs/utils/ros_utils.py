from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState

from baselines.agent.scara_arm.tree_urdf import treeFromFile # For KDL Jacobians
from PyKDL import Jacobian, Chain, ChainJntToJacSolver, JntArray # For KDL Jacobians

class MSG_INVALID_JOINT_NAMES_DIFFER(Exception):
    """Error object exclusively raised by _process_observations."""
    pass

def get_trajectory_message(action, joint_order):
    """
    Helper function.
    Wraps an action vector of joint angles into a JointTrajectory message.
    The velocities, accelerations, and effort do not control the arm motion
    """
    # Set up a trajectory message to publish.
    action_msg = JointTrajectory()
    # action_msg.joint_names = self.environment['joint_order']
    action_msg.joint_names = joint_order
    # Create a point to tell the robot to move to.
    target = JointTrajectoryPoint()
    action_float = [float(i) for i in action]
    target.positions = action_float

    # Package the single point into a trajectory of points with length 1.
    action_msg.points = [target]
    return action_msg

def process_observations(message, agent):
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
                raise MSG_INVALID_JOINT_NAMES_DIFFER
                print("Joints differ")
        return np.array(message.actual.positions) # + message.actual.velocities

def get_jacobians(state, scara_chain, jac_solver):
    """
    Produce a Jacobian from the urdf that maps from joint angles to x, y, z.
    This makes a 6x6 matrix from 6 joint angles to x, y, z and 3 angles.
    The angles are roll, pitch, and yaw (not Euler angles) and are not needed.
    Returns a repackaged Jacobian that is 3x6.
    """
    # Initialize a Jacobian for scara_chain.getNrOfJoints() joint angles by 3 cartesian coords and 3 orientation angles
    jacobian = Jacobian(scara_chain.getNrOfJoints())
    # Initialize a joint array for the present self.scara_chain.getNrOfJoints() joint angles.
    angles = JntArray(scara_chain.getNrOfJoints())
    # Construct the joint array from the most recent joint angles.
    for i in range(scara_chain.getNrOfJoints()):
        angles[i] = state[i]
    # Update the jacobian by solving for the given angles.observation_callback
    jac_solver.JntToJac(angles, jacobian)
    # Initialize a numpy array to store the Jacobian.
    J = np.array([[jacobian[i, j] for j in range(jacobian.columns())] for i in range(jacobian.rows())])
    # Only want the cartesian position, not Roll, Pitch, Yaw (RPY) Angles
    ee_jacobians = J
    return ee_jacobians

def get_ee_points_jacobians(ref_jacobian, ee_points, ref_rot):
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
                                        (0, 2, 1)).reshape(-1, scara_chain.getNrOfJoints())
    ee_points_jac_rot = np.tile(ref_jacobians_rot, (ee_points.shape[0], 1))
    return ee_points_jac_trans, ee_points_jac_rot

def get_ee_points_velocities(ref_jacobian, ee_points, ref_rot, joint_velocities):
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
