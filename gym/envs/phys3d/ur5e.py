"""Trains a ur5e robot arm to move its end effector to a sequence of targets.

The 6 joints have been placed faithfully to the actual joint locations of the
ur5e robot arm model.  Because brax does not yet support meshes, the capsule
locations for collider boundaries are only approximate.

See https://www.universal-robots.com/products/ur5-robot/ for more details.
"""

from typing import Tuple

import brax
import jumpy as jp
from brax import math

from gym.envs.phys3d.env import BraxEnv, BraxState


class Ur5e(BraxEnv):
    """Trains a UR5E robotic arm to touch a sequence of random targets."""

    def __init__(self, legacy_spring: bool = False, **kwargs):
        self.target_radius = 0.02
        self.target_distance = 0.5
        config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
        super().__init__(config=config, **kwargs)
        self.target_idx = self.sys.body.index["Target"]
        self.torso_idx = self.sys.body.index["wrist_3_link"]

    def brax_reset(self, rng: jp.ndarray) -> BraxState:
        qp = self.sys.default_qp()
        rng, target = self._random_target(rng)
        pos = jp.index_update(qp.pos, self.target_idx, target)
        qp = qp.replace(pos=pos)
        info = self.sys.info(qp)
        obs = self._get_obs(qp, info)
        reward, terminate, zero = jp.zeros(3)
        metrics = {
            "hits": zero,
            "weightedHits": zero,
            "movingToTarget": zero,
        }
        info = {"rng": rng}
        return BraxState(qp, obs, reward, terminate, metrics, info)

    def brax_step(self, state: BraxState, action: jp.ndarray) -> BraxState:
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info)

        # small reward for end effector moving towards target
        torso_delta = qp.pos[self.torso_idx] - state.qp.pos[self.torso_idx]
        target_rel = qp.pos[self.target_idx] - qp.pos[self.torso_idx]
        target_dist = jp.norm(target_rel)
        target_dir = target_rel / (1e-6 + target_dist)
        moving_to_target = 0.1 * jp.dot(torso_delta, target_dir)

        # big reward for reaching target
        target_hit = target_dist < self.target_radius
        target_hit = jp.where(target_hit, jp.float32(1), jp.float32(0))
        weighted_hit = target_hit

        reward = moving_to_target + weighted_hit

        state.metrics.update(
            hits=target_hit,
            weightedHits=weighted_hit,
            movingToTarget=moving_to_target,
        )

        # teleport any hit targets
        rng, target = self._random_target(state.info["rng"])
        target = jp.where(target_hit, target, qp.pos[self.target_idx])
        pos = jp.index_update(qp.pos, self.target_idx, target)
        qp = qp.replace(pos=pos)
        state.info.update(rng=rng)
        return state.replace(qp=qp, obs=obs, reward=reward)

    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        """Egocentric observation of target and arm body."""
        torso_fwd = math.rotate(jp.array([1.0, 0.0, 0.0]), qp.rot[self.torso_idx])
        torso_up = math.rotate(jp.array([0.0, 0.0, 1.0]), qp.rot[self.torso_idx])

        v_inv_rotate = jp.vmap(math.inv_rotate, include=(True, False))

        pos_local = qp.pos - qp.pos[self.torso_idx]
        pos_local = v_inv_rotate(pos_local, qp.rot[self.torso_idx])
        vel_local = v_inv_rotate(qp.vel, qp.rot[self.torso_idx])

        target_local = pos_local[self.target_idx]
        target_local_mag = jp.reshape(jp.norm(target_local), -1)
        target_local_dir = target_local / (1e-6 + target_local_mag)

        pos_local = jp.reshape(pos_local, -1)
        vel_local = jp.reshape(vel_local, -1)

        contact_mag = jp.sum(jp.square(info.contact.vel), axis=-1)
        contacts = jp.where(contact_mag > 0.00001, 1, 0)

        return jp.concatenate(
            [
                torso_fwd,
                torso_up,
                target_local_mag,
                target_local_dir,
                pos_local,
                vel_local,
                contacts,
            ]
        )

    def _random_target(self, rng: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        """Returns a target location in a random circle slightly above xy plane."""
        rng, rng1, rng2 = jp.random_split(rng, 3)
        dist = self.target_radius + self.target_distance * jp.random_uniform(rng1)
        ang = jp.pi * 2.0 * jp.random_uniform(rng2)
        target_x = dist * jp.cos(ang)
        target_y = dist * jp.sin(ang)
        target_z = 0.5
        target = jp.array([target_x, target_y, target_z]).transpose()
        return rng, target


_SYSTEM_CONFIG = """
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
    frozen {
      all: true
    }
  }
  bodies {
    name: "shoulder_link"
    colliders {
      position {
        y: 0.06682991981506348
      }
      rotation {
        x: 90.0
      }
      capsule {
        radius: 0.05945208668708801
        length: 0.13365983963012695
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "upper_arm_link"
    colliders {
      position {
        z: 0.21287038922309875
      }
      rotation {
      }
      capsule {
        radius: 0.05968618765473366
        length: 0.5446449518203735
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "forearm_link"
    colliders {
      position {
        z: 0.1851803958415985
      }
      rotation {
      }
      capsule {
        radius: 0.05584339052438736
        length: 0.48926496505737305
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "wrist_1_link"
    colliders {
      position {
        y: 0.10467606782913208
      }
      rotation {
        x: 90.0
      }
      capsule {
        radius: 0.038744933903217316
        length: 0.10467606782913208
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "wrist_2_link"
    colliders {
      position {
        z: 0.052344050258398056
      }
      rotation {
      }
      capsule {
        radius: 0.03879201412200928
        length: 0.10468810051679611
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "wrist_3_link"
    colliders {
      position {
        y: -0.04025782644748688
      }
      rotation {
        x: 90.0
      }
      capsule {
        radius: 0.01725015603005886
        length: 0.08051565289497375
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "Target"
    colliders {
      sphere {
        radius: 0.1
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
    frozen {
      all: true
    }
  }
  joints {
    name: "shoulder_pan_joint"
    parent: "floor"
    child: "shoulder_link"
    parent_offset {
      z: 0.163
    }
    child_offset {
    }
    rotation {
      y: -90.0
    }
    angular_damping: 50.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  joints {
    name: "shoulder_lift_joint"
    parent: "shoulder_link"
    child: "upper_arm_link"
    parent_offset {
      y: 0.138
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angular_damping: 50.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  joints {
    name: "elbow_joint"
    parent: "upper_arm_link"
    child: "forearm_link"
    parent_offset {
      y: -0.13
      z: 0.425
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angular_damping: 50.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  joints {
    name: "wrist_1_joint"
    parent: "forearm_link"
    child: "wrist_1_link"
    parent_offset {
      z: 0.3919999897480011
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angular_damping: 50.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  joints {
    name: "wrist_2_joint"
    parent: "wrist_1_link"
    child: "wrist_2_link"
    parent_offset {
      y: 0.12700000405311584
    }
    child_offset {
    }
    rotation {
      y: -90.0
    }
    angular_damping: 50.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  joints {
    name: "wrist_3_joint"
    parent: "wrist_2_link"
    child: "wrist_3_link"
    parent_offset {
      z: 0.1
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angular_damping: 50.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  actuators {
    name: "shoulder_pan_joint"
    joint: "shoulder_pan_joint"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "shoulder_lift_joint"
    joint: "shoulder_lift_joint"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "elbow_joint"
    joint: "elbow_joint"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "wrist_1_joint"
    joint: "wrist_1_joint"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "wrist_2_joint"
    joint: "wrist_2_joint"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "wrist_3_joint"
    joint: "wrist_3_joint"
    strength: 100.0
    torque {
    }
  }
  gravity {
    z: -9.81
  }
  collide_include {}
  angular_damping: -0.05
  dt: 0.02
  substeps: 8
  dynamics_mode: "pbd"
  """

_SYSTEM_CONFIG_SPRING = """
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
    frozen {
      all: true
    }
  }
  bodies {
    name: "shoulder_link"
    colliders {
      position {
        y: 0.06682991981506348
      }
      rotation {
        x: 90.0
      }
      capsule {
        radius: 0.05945208668708801
        length: 0.13365983963012695
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "upper_arm_link"
    colliders {
      position {
        z: 0.21287038922309875
      }
      rotation {
      }
      capsule {
        radius: 0.05968618765473366
        length: 0.5446449518203735
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "forearm_link"
    colliders {
      position {
        z: 0.1851803958415985
      }
      rotation {
      }
      capsule {
        radius: 0.05584339052438736
        length: 0.48926496505737305
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "wrist_1_link"
    colliders {
      position {
        y: 0.10467606782913208
      }
      rotation {
        x: 90.0
      }
      capsule {
        radius: 0.038744933903217316
        length: 0.10467606782913208
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "wrist_2_link"
    colliders {
      position {
        z: 0.052344050258398056
      }
      rotation {
      }
      capsule {
        radius: 0.03879201412200928
        length: 0.10468810051679611
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "wrist_3_link"
    colliders {
      position {
        y: -0.04025782644748688
      }
      rotation {
        x: 90.0
      }
      capsule {
        radius: 0.01725015603005886
        length: 0.08051565289497375
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "Target"
    colliders {
      sphere {
        radius: 0.1
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
    frozen {
      all: true
    }
  }
  joints {
    name: "shoulder_pan_joint"
    stiffness: 40000.0
    parent: "floor"
    child: "shoulder_link"
    parent_offset {
      z: 0.163
    }
    child_offset {
    }
    rotation {
      y: -90.0
    }
    angular_damping: 50.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  joints {
    name: "shoulder_lift_joint"
    stiffness: 40000.0
    parent: "shoulder_link"
    child: "upper_arm_link"
    parent_offset {
      y: 0.138
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angular_damping: 50.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  joints {
    name: "elbow_joint"
    stiffness: 40000.0
    parent: "upper_arm_link"
    child: "forearm_link"
    parent_offset {
      y: -0.13
      z: 0.425
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angular_damping: 50.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  joints {
    name: "wrist_1_joint"
    stiffness: 40000.0
    parent: "forearm_link"
    child: "wrist_1_link"
    parent_offset {
      z: 0.3919999897480011
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angular_damping: 50.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  joints {
    name: "wrist_2_joint"
    stiffness: 40000.0
    parent: "wrist_1_link"
    child: "wrist_2_link"
    parent_offset {
      y: 0.12700000405311584
    }
    child_offset {
    }
    rotation {
      y: -90.0
    }
    angular_damping: 50.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  joints {
    name: "wrist_3_joint"
    stiffness: 40000.0
    parent: "wrist_2_link"
    child: "wrist_3_link"
    parent_offset {
      z: 0.1
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angular_damping: 50.0
    angle_limit {
      min: -360.0
      max: 360.0
    }
  }
  actuators {
    name: "shoulder_pan_joint"
    joint: "shoulder_pan_joint"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "shoulder_lift_joint"
    joint: "shoulder_lift_joint"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "elbow_joint"
    joint: "elbow_joint"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "wrist_1_joint"
    joint: "wrist_1_joint"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "wrist_2_joint"
    joint: "wrist_2_joint"
    strength: 100.0
    torque {
    }
  }
  actuators {
    name: "wrist_3_joint"
    joint: "wrist_3_joint"
    strength: 100.0
    torque {
    }
  }
  gravity {
    z: -9.81
  }
  collide_include {}
  angular_damping: -0.05
  baumgarte_erp: 0.1
  dt: 0.02
  substeps: 8
  dynamics_mode: "legacy_spring"
  """
