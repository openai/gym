"""Trains a humanoid to run in the +x direction."""

import brax
import jumpy as jp
from brax.physics import bodies

from gym.envs.phys3d.env import BraxEnv, BraxState


class Humanoid(BraxEnv):
    """Trains a humanoid to run in the +x direction."""

    def __init__(self, legacy_spring: bool = False, **kwargs):
        config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
        super().__init__(config=config, **kwargs)
        body = bodies.Body(self.sys.config)
        body = jp.take(body, body.idx[:-1])  # skip the floor body
        self.mass = body.mass.reshape(-1, 1)
        self.inertia = body.inertia
        self.inertia_matrix = jp.array([jp.diag(a) for a in self.inertia])

    def brax_reset(self, rng: jp.ndarray) -> BraxState:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jp.random_split(rng, 3)
        qpos = self.sys.default_angle() + jp.random_uniform(
            rng1, (self.sys.num_joint_dof,), -0.01, 0.01
        )
        qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -0.01, 0.01)
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        info = self.sys.info(qp)
        obs = self._get_obs(qp, info, jp.zeros(self.action_size))
        reward, terminate, zero = jp.zeros(3)
        metrics = {
            "reward_linvel": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "reward_impact": zero,
        }
        return BraxState(qp, obs, reward, terminate, metrics)

    def brax_step(self, state: BraxState, action: jp.ndarray) -> BraxState:
        """Run one timestep of the environment's dynamics."""
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info, action)

        pos_before = state.qp.pos[:-1]  # ignore floor at last index
        pos_after = qp.pos[:-1]  # ignore floor at last index
        com_before = jp.sum(pos_before * self.mass, axis=0) / jp.sum(self.mass)
        com_after = jp.sum(pos_after * self.mass, axis=0) / jp.sum(self.mass)
        lin_vel_cost = 1.25 * (com_after[0] - com_before[0]) / self.sys.config.dt
        quad_ctrl_cost = 0.01 * jp.sum(jp.square(action))
        # can ignore contact cost, see: https://github.com/openai/gym/issues/1541
        quad_impact_cost = jp.float32(0)
        alive_bonus = jp.float32(5)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

        terminate = jp.where(qp.pos[0, 2] < 0.8, jp.float32(1), jp.float32(0))
        terminate = jp.where(qp.pos[0, 2] > 2.1, jp.float32(1), terminate)
        state.metrics.update(
            reward_linvel=lin_vel_cost,
            reward_quadctrl=quad_ctrl_cost,
            reward_alive=alive_bonus,
            reward_impact=quad_impact_cost,
        )

        return state.replace(qp=qp, obs=obs, reward=reward, terminate=terminate)

    def _get_obs(self, qp: brax.QP, info: brax.Info, action: jp.ndarray) -> jp.ndarray:
        """Observe humanoid body position, velocities, and angles."""
        # some pre-processing to pull joint angles and velocities
        joint_obs = [j.angle_vel(qp) for j in self.sys.joints]

        # qpos:
        # Z of the torso (1,)
        # orientation of the torso as quaternion (4,)
        # joint angles (8,)
        joint_angles = [jp.array(j[0]).reshape(-1) for j in joint_obs]
        qpos = [qp.pos[0, 2:], qp.rot[0]] + joint_angles

        # qvel:
        # velocity of the torso (3,)
        # angular velocity of the torso (3,)
        # joint angle velocities (8,)
        joint_velocities = [jp.array(j[1]).reshape(-1) for j in joint_obs]
        qvel = [qp.vel[0], qp.ang[0]] + joint_velocities

        # actuator forces
        qfrc_actuator = []
        for act in self.sys.actuators:
            torque = jp.take(action, act.act_index)
            torque = torque.reshape(torque.shape[:-2] + (-1,))
            torque *= jp.repeat(act.strength, act.act_index.shape[-1])
            qfrc_actuator.append(torque)

        # external contact forces:
        # delta velocity (3,), delta ang (3,) * num bodies in the system
        cfrc_ext = [info.contact.vel, info.contact.ang]
        # flatten bottom dimension
        cfrc_ext = [x.reshape(x.shape[:-2] + (-1,)) for x in cfrc_ext]

        # center of mass obs:
        body_pos = qp.pos[:-1]  # ignore floor at last index
        body_vel = qp.vel[:-1]  # ignore floor at last index

        com_vec = jp.sum(body_pos * self.mass, axis=0) / jp.sum(self.mass)
        com_vel = body_vel * self.mass / jp.sum(self.mass)

        v_outer = jp.vmap(lambda a: jp.outer(a, a))
        v_cross = jp.vmap(jp.cross)

        disp_vec = body_pos - com_vec
        com_inert = self.inertia_matrix + self.mass.reshape((11, 1, 1)) * (
            (jp.norm(disp_vec, axis=1) ** 2.0).reshape((11, 1, 1))
            * jp.stack([jp.eye(3)] * 11)
            - v_outer(disp_vec)
        )

        cinert = [com_inert.reshape(-1)]

        square_disp = (1e-7 + (jp.norm(disp_vec, axis=1) ** 2.0)).reshape((11, 1))
        com_angular_vel = v_cross(disp_vec, body_vel) / square_disp
        cvel = [com_vel.reshape(-1), com_angular_vel.reshape(-1)]

        return jp.concatenate(qpos + qvel + cinert + cvel + qfrc_actuator + cfrc_ext)


_SYSTEM_CONFIG = """
  bodies {
    name: "torso"
    colliders {
      position {
      }
      rotation {
        x: -90.0
      }
      capsule {
        radius: 0.07
        length: 0.28
      }
    }
    colliders {
      position {
        z: 0.19
      }
      capsule {
        radius: 0.09
        length: 0.18
      }
    }
    colliders {
      position {
        x: -0.01
        z: -0.12
      }
      rotation {
        x: -90.0
      }
      capsule {
        radius: 0.06
        length: 0.24
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 8.907463
  }
  bodies {
    name: "lwaist"
    colliders {
      position {
      }
      rotation {
        x: -90.0
      }
      capsule {
        radius: 0.06
        length: 0.24
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 2.2619467
  }
  bodies {
    name: "pelvis"
    colliders {
      position {
        x: -0.02
      }
      rotation {
        x: -90.0
      }
      capsule {
        radius: 0.09
        length: 0.32
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 6.6161942
  }
  bodies {
    name: "right_thigh"
    colliders {
      position {
        y: 0.005
        z: -0.17
      }
      rotation {
        x: -178.31532
      }
      capsule {
        radius: 0.06
        length: 0.46014702
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 4.751751
  }
  bodies {
    name: "right_shin"
    colliders {
      position {
        z: -0.15
      }
      rotation {
        x: -180.0
      }
      capsule {
        radius: 0.049
        length: 0.398
        end: -1
      }
    }
    colliders {
      position {
        z: -0.35
      }
      capsule {
        radius: 0.075
        length: 0.15
        end: 1
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 4.5228419
  }
  bodies {
    name: "left_thigh"
    colliders {
      position {
        y: -0.005
        z: -0.17
      }
      rotation {
        x: 178.31532
      }
      capsule {
        radius: 0.06
        length: 0.46014702
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 4.751751
  }
  bodies {
    name: "left_shin"
    colliders {
      position {
        z: -0.15
      }
      rotation {
        x: -180.0
      }
      capsule {
        radius: 0.049
        length: 0.398
        end: -1
      }
    }
    colliders {
      position {
        z: -0.35
      }
      capsule {
        radius: 0.075
        length: 0.15
        end: 1
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 4.5228419
  }
  bodies {
    name: "right_upper_arm"
    colliders {
      position {
        x: 0.08
        y: -0.08
        z: -0.08
      }
      rotation {
        x: 135.0
        y: 35.26439
        z: -75.0
      }
      capsule {
        radius: 0.04
        length: 0.35712814
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.6610805
  }
  bodies {
    name: "right_lower_arm"
    colliders {
      position {
        x: 0.09
        y: 0.09
        z: 0.09
      }
      rotation {
        x: -45.0
        y: 35.26439
        z: 15.0
      }
      capsule {
        radius: 0.031
        length: 0.33912814
      }
    }
    colliders {
      position {
        x: 0.18
        y: 0.18
        z: 0.18
      }
      capsule {
        radius: 0.04
        length: 0.08
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.2295402
  }
  bodies {
    name: "left_upper_arm"
    colliders {
      position {
        x: 0.08
        y: 0.08
        z: -0.08
      }
      rotation {
        x: -135.0
        y: 35.26439
        z: 75.0
      }
      capsule {
        radius: 0.04
        length: 0.35712814
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.6610805
  }
  bodies {
    name: "left_lower_arm"
    colliders {
      position {
        x: 0.09
        y: -0.09
        z: 0.09
      }
      rotation {
        x: 45.0
        y: 35.26439
        z: -15.0
      }
      capsule {
        radius: 0.031
        length: 0.33912814
      }
    }
    colliders {
      position {
        x: 0.18
        y: -0.18
        z: 0.18
      }
      capsule {
        radius: 0.04
        length: 0.08
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.2295402
  }
  bodies {
    name: "floor"
    colliders {
      plane {
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
    frozen { all: true }
  }
  joints {
    name: "abdomen_z"
    parent: "torso"
    child: "lwaist"
    parent_offset {
      x: -0.01
      z: -0.195
    }
    child_offset {
      z: 0.065
    }
    rotation {
      y: -90.0
    }
    angle_limit {
      min: -45.0
      max: 45.0
    }
    angle_limit {
      min: -65.0
      max: 30.0
    }
      angle_limit {
      min: 0.0
      max: 0.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "abdomen_x"
    parent: "lwaist"
    child: "pelvis"
    parent_offset {
      z: -0.065
    }
    child_offset {
      z: 0.1
    }
    rotation {
      x: 90.0
    }
    angle_limit {
      min: -35.0
      max: 35.0
    }
      angle_limit {
      min: 0.0
      max: 0.0
    }
      angle_limit {
      min: 0.0
      max: 0.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "right_hip_x"
    parent: "pelvis"
    child: "right_thigh"
    parent_offset {
      y: -0.1
      z: -0.04
    }
    child_offset {
    }
    rotation {
    }
    angle_limit {
      min: -10.0
      max: 10.0
    }
    angle_limit {
      min: -30.0
      max: 70.0
    }
    angle_limit {
      min: -10.0
      max: 10.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "right_knee"
    parent: "right_thigh"
    child: "right_shin"
    parent_offset {
      y: 0.01
      z: -0.383
    }
    child_offset {
      z: 0.02
    }
    rotation {
      z: -90.0
    }
    angle_limit {
      min: -160.0
      max: -2.0
    }
      angle_limit {
      min: 0.0
      max: 0.0
    }
      angle_limit {
      min: 0.0
      max: 0.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "left_hip_x"
    parent: "pelvis"
    child: "left_thigh"
    parent_offset {
      y: 0.1
      z: -0.04
    }
    child_offset {
    }
    angle_limit {
      min: -10.0
      max: 10.0
    }
    angle_limit {
      min: -30.0
      max: 70.0
    }
    angle_limit {
      min: -10.0
      max: 10.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "left_knee"
    parent: "left_thigh"
    child: "left_shin"
    parent_offset {
      y: -0.01
      z: -0.383
    }
    child_offset {
      z: 0.02
    }
    rotation {
      z: -90.0
    }
    angle_limit {
      min: -160.0
      max: -2.0
    }
      angle_limit {
      min: 0.0
      max: 0.0
    }
      angle_limit {
      min: 0.0
      max: 0.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "right_shoulder1"
    parent: "torso"
    child: "right_upper_arm"
    parent_offset {
      y: -0.17
      z: 0.06
    }
    child_offset {
    }
    rotation {
      x: 135.0
      y: 35.26439
    }
    angle_limit {
      min: -85.0
      max: 60.0
    }
    angle_limit {
      min: -70.0
      max: 50.0
    }
      angle_limit {
      min: 0.0
      max: 0.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "right_elbow"
    parent: "right_upper_arm"
    child: "right_lower_arm"
    parent_offset {
      x: 0.18
      y: -0.18
      z: -0.18
    }
    child_offset {
    }
    rotation {
      x: 135.0
      z: 90.0
    }
    angle_limit {
      min: -90.0
      max: 50.0
    }
      angle_limit {
      min: 0.0
      max: 0.0
    }
      angle_limit {
      min: 0.0
      max: 0.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "left_shoulder1"
    parent: "torso"
    child: "left_upper_arm"
    parent_offset {
      y: 0.17
      z: 0.06
    }
    child_offset {
    }
    rotation {
      x: 45.0
      y: -35.26439
    }
    angle_limit {
      min: -60.0
      max: 85.0
    }
    angle_limit {
      min: -50.0
      max: 70.0
    }
      angle_limit {
      min: 0.0
      max: 0.0
    }
    angular_damping: 30.0
  }
  joints {
    name: "left_elbow"
    parent: "left_upper_arm"
    child: "left_lower_arm"
    parent_offset {
      x: 0.18
      y: 0.18
      z: -0.18
    }
    child_offset {
    }
    rotation {
      x: 45.0
      z: -90.0
    }
    angle_limit {
      min: -90.0
      max: 50.0
    }
      angle_limit {
      min: 0.0
      max: 0.0
    }
      angle_limit {
      min: 0.0
      max: 0.0
    }
    angular_damping: 30.0
  }
  actuators {
    name: "abdomen_z"
    joint: "abdomen_z"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "abdomen_x"
    joint: "abdomen_x"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "right_hip_x"
    joint: "right_hip_x"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "right_knee"
    joint: "right_knee"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "left_hip_x"
    joint: "left_hip_x"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "left_knee"
    joint: "left_knee"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "right_shoulder1"
    joint: "right_shoulder1"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "right_elbow"
    joint: "right_elbow"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "left_shoulder1"
    joint: "left_shoulder1"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "left_elbow"
    joint: "left_elbow"
    strength: 100.0
    torque {
    }
  }
  collide_include {
    first: "floor"
    second: "left_shin"
  }
  collide_include {
    first: "floor"
    second: "right_shin"
  }
  defaults {
    angles {
      name: "left_knee"
      angle { x: -25. y: 0 z: 0 }
    }
    angles {
      name: "right_knee"
      angle { x: -25. y: 0 z: 0 }
    }
  }
  friction: 1.0
  gravity {
    z: -9.81
  }
  angular_damping: -0.05
  dt: 0.015
  substeps: 8
  dynamics_mode: "pbd"
  """

_SYSTEM_CONFIG_SPRING = """
  bodies {
    name: "torso"
    colliders {
      position {
      }
      rotation {
        x: -90.0
      }
      capsule {
        radius: 0.07
        length: 0.28
      }
    }
    colliders {
      position {
        z: 0.19
      }
      capsule {
        radius: 0.09
        length: 0.18
      }
    }
    colliders {
      position {
        x: -0.01
        z: -0.12
      }
      rotation {
        x: -90.0
      }
      capsule {
        radius: 0.06
        length: 0.24
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 8.907463
  }
  bodies {
    name: "lwaist"
    colliders {
      position {
      }
      rotation {
        x: -90.0
      }
      capsule {
        radius: 0.06
        length: 0.24
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 2.2619467
  }
  bodies {
    name: "pelvis"
    colliders {
      position {
        x: -0.02
      }
      rotation {
        x: -90.0
      }
      capsule {
        radius: 0.09
        length: 0.32
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 6.6161942
  }
  bodies {
    name: "right_thigh"
    colliders {
      position {
        y: 0.005
        z: -0.17
      }
      rotation {
        x: -178.31532
      }
      capsule {
        radius: 0.06
        length: 0.46014702
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 4.751751
  }
  bodies {
    name: "right_shin"
    colliders {
      position {
        z: -0.15
      }
      rotation {
        x: -180.0
      }
      capsule {
        radius: 0.049
        length: 0.398
        end: -1
      }
    }
    colliders {
      position {
        z: -0.35
      }
      capsule {
        radius: 0.075
        length: 0.15
        end: 1
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 4.5228419
  }
  bodies {
    name: "left_thigh"
    colliders {
      position {
        y: -0.005
        z: -0.17
      }
      rotation {
        x: 178.31532
      }
      capsule {
        radius: 0.06
        length: 0.46014702
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 4.751751
  }
  bodies {
    name: "left_shin"
    colliders {
      position {
        z: -0.15
      }
      rotation {
        x: -180.0
      }
      capsule {
        radius: 0.049
        length: 0.398
        end: -1
      }
    }
    colliders {
      position {
        z: -0.35
      }
      capsule {
        radius: 0.075
        length: 0.15
        end: 1
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 4.5228419
  }
  bodies {
    name: "right_upper_arm"
    colliders {
      position {
        x: 0.08
        y: -0.08
        z: -0.08
      }
      rotation {
        x: 135.0
        y: 35.26439
        z: -75.0
      }
      capsule {
        radius: 0.04
        length: 0.35712814
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.6610805
  }
  bodies {
    name: "right_lower_arm"
    colliders {
      position {
        x: 0.09
        y: 0.09
        z: 0.09
      }
      rotation {
        x: -45.0
        y: 35.26439
        z: 15.0
      }
      capsule {
        radius: 0.031
        length: 0.33912814
      }
    }
    colliders {
      position {
        x: 0.18
        y: 0.18
        z: 0.18
      }
      capsule {
        radius: 0.04
        length: 0.08
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.2295402
  }
  bodies {
    name: "left_upper_arm"
    colliders {
      position {
        x: 0.08
        y: 0.08
        z: -0.08
      }
      rotation {
        x: -135.0
        y: 35.26439
        z: 75.0
      }
      capsule {
        radius: 0.04
        length: 0.35712814
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.6610805
  }
  bodies {
    name: "left_lower_arm"
    colliders {
      position {
        x: 0.09
        y: -0.09
        z: 0.09
      }
      rotation {
        x: 45.0
        y: 35.26439
        z: -15.0
      }
      capsule {
        radius: 0.031
        length: 0.33912814
      }
    }
    colliders {
      position {
        x: 0.18
        y: -0.18
        z: 0.18
      }
      capsule {
        radius: 0.04
        length: 0.08
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.2295402
  }
  bodies {
    name: "floor"
    colliders {
      plane {
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
    frozen { all: true }
  }
  joints {
    name: "abdomen_z"
    stiffness: 27000
    parent: "torso"
    child: "lwaist"
    parent_offset {
      x: -0.01
      z: -0.195
    }
    child_offset {
      z: 0.065
    }
    rotation {
      y: -90.0
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -45.0
      max: 45.0
    }
    angle_limit {
      min: -75.0
      max: 30.0
    }
  }
  joints {
    name: "abdomen_x"
    stiffness: 27000
    parent: "lwaist"
    child: "pelvis"
    parent_offset {
      z: -0.065
    }
    child_offset {
      z: 0.1
    }
    rotation {
      x: 90.0
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -35.0
      max: 35.0
    }
  }
  joints {
    name: "right_hip_x"
    stiffness: 27000
    parent: "pelvis"
    child: "right_thigh"
    parent_offset {
      y: -0.1
      z: -0.04
    }
    child_offset {
    }
    rotation {
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -10.0
      max: 10.0
    }
    angle_limit {
      min: -30.0
      max: 70.0
    }
    angle_limit {
      min: -10.0
      max: 10.0
    }
  }
  joints {
    name: "right_knee"
    stiffness: 27000
    parent: "right_thigh"
    child: "right_shin"
    parent_offset {
      y: 0.01
      z: -0.383
    }
    child_offset {
      z: 0.02
    }
    rotation {
      z: -90.0
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -160.0
      max: -2.0
    }
  }
  joints {
    name: "left_hip_x"
    stiffness: 27000
    parent: "pelvis"
    child: "left_thigh"
    parent_offset {
      y: 0.1
      z: -0.04
    }
    child_offset {
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -10.0
      max: 10.0
    }
    angle_limit {
      min: -30.0
      max: 70.0
    }
    angle_limit {
      min: -10.0
      max: 10.0
    }
  }
  joints {
    name: "left_knee"
    stiffness: 27000
    parent: "left_thigh"
    child: "left_shin"
    parent_offset {
      y: -0.01
      z: -0.383
    }
    child_offset {
      z: 0.02
    }
    rotation {
      z: -90.0
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -160.0
      max: -2.0
    }
  }
  joints {
    name: "right_shoulder1"
    stiffness: 27000
    parent: "torso"
    child: "right_upper_arm"
    parent_offset {
      y: -0.17
      z: 0.06
    }
    child_offset {
    }
    rotation {
      x: 135.0
      y: 35.26439
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -85.0
      max: 60.0
    }
    angle_limit {
      min: -85.0
      max: 60.0
    }
  }
  joints {
    name: "right_elbow"
    stiffness: 27000
    parent: "right_upper_arm"
    child: "right_lower_arm"
    parent_offset {
      x: 0.18
      y: -0.18
      z: -0.18
    }
    child_offset {
    }
    rotation {
      x: 135.0
      z: 90.0
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -90.0
      max: 50.0
    }
  }
  joints {
    name: "left_shoulder1"
    stiffness: 27000
    parent: "torso"
    child: "left_upper_arm"
    parent_offset {
      y: 0.17
      z: 0.06
    }
    child_offset {
    }
    rotation {
      x: 45.0
      y: -35.26439
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -60.0
      max: 85.0
    }
    angle_limit {
      min: -60.0
      max: 85.0
    }
  }
  joints {
    name: "left_elbow"
    stiffness: 27000
    parent: "left_upper_arm"
    child: "left_lower_arm"
    parent_offset {
      x: 0.18
      y: 0.18
      z: -0.18
    }
    child_offset {
    }
    rotation {
      x: 45.0
      z: -90.0
    }
    angular_damping: 30
    spring_damping: 80
    limit_strength: 2500
    angle_limit {
      min: -90.0
      max: 50.0
    }
  }
  actuators {
    name: "abdomen_z"
    joint: "abdomen_z"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "abdomen_x"
    joint: "abdomen_x"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "right_hip_x"
    joint: "right_hip_x"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "right_knee"
    joint: "right_knee"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "left_hip_x"
    joint: "left_hip_x"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "left_knee"
    joint: "left_knee"
    strength: 350.0
    torque {
    }
  }
  actuators {
    name: "right_shoulder1"
    joint: "right_shoulder1"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "right_elbow"
    joint: "right_elbow"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "left_shoulder1"
    joint: "left_shoulder1"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "left_elbow"
    joint: "left_elbow"
    strength: 100.0
    torque {
    }
  }
  collide_include {
    first: "floor"
    second: "left_shin"
  }
  collide_include {
    first: "floor"
    second: "right_shin"
  }
  defaults {
    angles {
      name: "left_knee"
      angle { x: -25. y: 0 z: 0 }
    }
    angles {
      name: "right_knee"
      angle { x: -25. y: 0 z: 0 }
    }
  }
  friction: 1.0
  gravity {
    z: -9.81
  }
  angular_damping: -0.05
  baumgarte_erp: 0.1
  dt: 0.015
  substeps: 8
  dynamics_mode: "legacy_spring"
  """
