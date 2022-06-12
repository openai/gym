"""Trains a claw hand to grasp an object and move it to targets."""

from typing import Tuple

import brax
import jumpy as jp
from brax import math

from gym.envs.jax_env import JaxEnv, JaxState


class Grasp(JaxEnv):
    """Grasp trains an agent to pick up an object.

    Grasp observes three bodies: 'Hand', 'Object', and 'Target'.
    When Object reaches Target, the agent is rewarded.
    """

    def __init__(self, legacy_spring: bool = False, **kwargs):
        self.target_radius = 1.1
        self.target_distance = 10.0
        self.target_height = 8.0

        config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
        super().__init__(config=config, **kwargs)

        self.object_idx = self.sys.body.index["Object"]
        self.target_idx = self.sys.body.index["Target"]
        self.hand_idx = self.sys.body.index["HandThumbProximal"]
        self.palm_idx = self.sys.body.index["HandPalm"]

        # map the [-1, 1] action space into a valid angle for the actuators
        limits = []
        for joint in self.sys.config.joints:
            for limit in joint.angle_limit:
                limits.append((limit.min, limit.max))
        self._min_act = jp.array([limit[0] for limit in limits])
        self._range_act = jp.array([limit[1] - limit[0] for limit in limits])

        # add limits for the translational motion of the hand base
        self._min_act = jp.concatenate([self._min_act, jp.array([-10, -10, 3.5])])
        self._range_act = jp.concatenate([self._range_act, jp.array([20, 20, 10])])

    def internal_reset(self, rng: jp.ndarray) -> JaxState:
        qp = self.sys.default_qp()
        # rng, target = self._random_target(rng)
        # pos = qp.pos.at[self.target_idx].set(target)
        # qp = dataclasses.replace(qp, pos=pos)
        info = self.sys.info(qp)
        obs = self._get_obs(qp, info)
        reward, terminate, zero = jp.zeros(3)
        metrics = {
            "hits": zero,
            "touchingObject": zero,
            "movingToObject": zero,
            "movingObjectToTarget": zero,
            "closeToObject": zero,
        }
        info = {"rng": rng}
        return JaxState(qp, obs, reward, terminate, metrics, info)

    def internal_step(self, state: JaxState, action: jp.ndarray) -> JaxState:
        # actuate the palm
        action = self._min_act + self._range_act * ((action + 1) / 2.0)
        target_pos = action[-3:]
        palm_pos = state.qp.pos[self.palm_idx]
        norm = jp.norm(target_pos - palm_pos)
        # make sure hand doesn't move too fast
        scale = jp.where(norm > 2.0, 2.0 / norm, 1.0)
        palm_pos = palm_pos + scale * (target_pos - palm_pos) * 0.15
        pos = state.qp.pos
        pos = jp.index_update(pos, self.palm_idx, palm_pos)
        qp = state.qp.replace(pos=pos)

        # do the rest of the physics update
        qp, info = self.sys.step(qp, action)
        obs = self._get_obs(qp, info)

        # small reward for moving in right direction
        object_pos = qp.pos[self.object_idx]
        hand_pos = qp.pos[self.palm_idx]
        hand_vel = qp.vel[self.hand_idx]
        object_rel = object_pos - hand_pos
        object_dist = jp.norm(object_rel)
        planar_object_dist = jp.norm(object_rel * jp.array([1.0, 1.0, 0.0]))
        object_dir = object_rel / (1e-6 + object_dist.reshape(-1))

        moving_to_object = 0.1 * self.sys.config.dt * jp.dot(hand_vel, object_dir)
        close_to_object = 0.1 * self.sys.config.dt * 1.0 / (1.0 + planar_object_dist)

        # large reward for object moving to target
        target_pos = qp.pos[self.target_idx]
        object_vel = qp.vel[self.object_idx]
        target_rel = target_pos - object_pos

        target_dist = jp.norm(target_rel)
        target_dir = target_rel / (1e-6 + target_dist)
        moving_to_target = 1.5 * self.sys.config.dt * jp.dot(object_vel, target_dir)

        # small reward for touching object
        contact_mag = jp.sum(jp.square(info.contact.vel), axis=-1)
        contacts = jp.where(contact_mag > 0.00001, 1, 0)
        touching_object = (
            0.2
            * self.sys.config.dt
            * (contacts[3] + contacts[9] + contacts[12] + contacts[15])
        )

        # big reward for reaching target
        target_hit = jp.where(target_dist < self.target_radius, 1.0, 0.0)

        reward = (
            moving_to_object
            + close_to_object
            + touching_object
            + 5.0 * target_hit
            + moving_to_target
        )

        state.metrics.update(
            hits=target_hit,
            touchingObject=touching_object,
            movingToObject=moving_to_object,
            movingObjectToTarget=moving_to_target,
            closeToObject=close_to_object,
        )

        # teleport any hit targets
        rng, target = self._random_target(state.info["rng"])
        target = jp.where(target_hit, target, qp.pos[self.target_idx])
        pos = jp.index_update(qp.pos, self.target_idx, target)
        qp = qp.replace(pos=pos)
        state.info.update(rng=rng)
        return state.replace(qp=qp, obs=obs, reward=reward)

    @property
    def action_size(self) -> int:
        return super().action_size + 3  # 3 extra actions for translating

    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        """Egocentric observation of target, object, and hand."""

        v_inv_rotate = jp.vmap(math.inv_rotate, include=(True, False))

        pos_local = qp.pos - qp.pos[self.palm_idx]
        pos_local = v_inv_rotate(pos_local, qp.rot[self.palm_idx])
        vel_local = v_inv_rotate(qp.vel, qp.rot[self.palm_idx])

        object_local = pos_local[self.object_idx]
        object_local_mag = jp.norm(object_local).reshape(-1)
        object_local_dir = object_local / (1e-6 + object_local_mag)

        # target specific obs
        hand_to_obj = qp.pos[self.object_idx] - qp.pos[self.palm_idx]
        hand_to_obj_mag = jp.norm(hand_to_obj)
        hand_to_obj_dir = hand_to_obj / (1e-6 + hand_to_obj_mag)
        hand_vel = qp.vel[self.hand_idx]
        heading_to_obj = jp.dot(hand_to_obj_dir, hand_vel).reshape(-1)

        target_local = pos_local[self.target_idx]
        target_local_mag = jp.norm(target_local).reshape(-1)
        target_local_dir = target_local / (1e-6 + target_local_mag)

        # object to target
        obj_to_target = qp.pos[self.target_idx] - qp.pos[self.object_idx]
        obj_to_target_mag = jp.norm(obj_to_target).reshape(-1)
        obj_to_target_dir = obj_to_target / (1e-6 + obj_to_target_mag)

        object_vel = qp.vel[self.object_idx]
        obj_heading_to_target = jp.dot(obj_to_target_dir, object_vel).reshape(-1)

        pos_local = pos_local.reshape(-1)
        vel_local = vel_local.reshape(-1)

        contact_mag = jp.sum(jp.square(info.contact.vel), axis=-1)
        contacts = jp.where(contact_mag > 0.00001, 1, 0)

        return jp.concatenate(
            [
                object_local_mag,
                object_local_dir,
                target_local_mag,
                target_local_dir,
                pos_local,
                vel_local,
                hand_to_obj,
                hand_vel,
                heading_to_obj,
                obj_to_target_mag,
                obj_to_target_dir,
                obj_heading_to_target,
                contacts,
            ]
        )

    def _random_target(self, rng: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        """Returns new random target locations in a random circle on xz plane."""
        rng, rng1, rng2, rng3 = jp.random_split(rng, 4)
        dist = self.target_radius + self.target_distance * jp.random_uniform(rng1)
        ang = jp.pi * 2.0 * jp.random_uniform(rng2)
        target_x = dist * jp.cos(ang)
        target_y = dist * jp.sin(ang)
        target_z = self.target_height * jp.random_uniform(rng3)
        target = jp.array([target_x, target_y, target_z]).transpose()
        return rng, target


_SYSTEM_CONFIG = """
  bodies {
    name: "Ground"
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
  bodies {
    name: "Object"
    colliders {
      capsule {
        radius: 1.0
        length: 2.02
      }
      rotation { x: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbProximal"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { x: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbDistal"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { x: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbMiddle"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { x: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandPalm"
    colliders {
      capsule {
        radius: 1.5
        length: 3.1
      }
      rotation { x: 120 }
    }
    inertia {
      x: 5000.0350000858306885
      y: 10000.7224998474121094
      z: 10000.7224998474121094
    }
    mass: 1.0
    frozen { all : true }
  }
  bodies {
    name: "HandThumbProximalTwo"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { x: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbMiddleTwo"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { x: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbDistalTwo"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { x: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbProximalThree"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { y: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbMiddleThree"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { y: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbDistalThree"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { y: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbProximalFour"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { y: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbMiddleFour"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { y: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbDistalFour"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { y: 90 }
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
    colliders { sphere { radius: 1.1 }}
    frozen { all: true }
  }
  joints {
    name: "HandThumbMiddle_HandThumbDistal"
    angular_damping: 50.0
    parent_offset {
      y: 0.75
    }
    child_offset {
      y: -0.75
    }
    parent: "HandThumbMiddle"
    child: "HandThumbDistal"

    angle_limit { min: -90.0 }
    angle_limit { min: -0.1 max: 0.1 }
    angle_limit { min: -0.1 max: 0.1 }
  }
  joints {
    name: "HandThumbProximal_HandThumbMiddle"
    angular_damping: 50.0
    parent_offset {
      y: 0.75
    }
    child_offset {
      y: -0.75
    }
    parent: "HandThumbProximal"
    child: "HandThumbMiddle"

    angle_limit { min: -90.0 }
    angle_limit { min: -0.1 max: 0.1 }
    angle_limit { min: -0.1 max: 0.1 }
  }
  joints {
    name: "HandPalm_HandThumbProximal"
    angular_damping: 50.0
    parent_offset {
      y: 1.5
      z: -.5
    }
    child_offset {
      y: -0.75
    }
    parent: "HandPalm"
    child: "HandThumbProximal"

    angle_limit { min: -45 max: 45 }
    angle_limit { min: -45 max: 45 }
    angle_limit { min: -0.1 max: 0.1 }
  }
  joints {
    name: "HandThumbMiddleTwo_HandThumbDistalTwo"
    angular_damping: 50.0
    parent_offset {
      y: -0.75
    }
    child_offset {
      y: 0.75
    }
    parent: "HandThumbMiddleTwo"
    child: "HandThumbDistalTwo"

    angle_limit { min: 0 max: 90 }
    angle_limit { min: -0.1 max: 0.1 }
    angle_limit { min: -0.1 max: 0.1 }

  }
  joints {
    name: "HandThumbProximalTwo_HandThumbMiddleTwo"
    angular_damping: 50.0
    parent_offset {
      y: -0.75
    }
    child_offset {
      y: 0.75
    }
    parent: "HandThumbProximalTwo"
    child: "HandThumbMiddleTwo"

    angle_limit { min: 0 max: 90 }
    angle_limit { min: -0.1 max: 0.1 }
    angle_limit { min: -0.1 max: 0.1 }
  }
  joints {
    name: "HandPalm_HandThumbProximalTwo"
    angular_damping: 50.0
    parent_offset {
      y: -1.5
      z: -.5
    }
    child_offset {
      y: 0.75
    }
    parent: "HandPalm"
    child: "HandThumbProximalTwo"

    angle_limit { min: -45 max: 45 }
    angle_limit { min: -45 max: 45 }
    angle_limit { min: -0.1 max: 0.1 }
  }
  joints {
    name: "HandThumbMiddleThree_HandThumbDistalThree"
    angular_damping: 50.0
    parent_offset {
      x: -0.75
    }
    child_offset {
      x: 0.75
    }
    parent: "HandThumbMiddleThree"
    child: "HandThumbDistalThree"

    rotation { z: -90 }
    angle_limit { min: 0 max: 90 }
    angle_limit { min: -0.1 max: 0.1 }
    angle_limit { min: -0.1 max: 0.1 }
  }
  joints {
    name: "HandThumbProximalThree_HandThumbMiddleThree"
    angular_damping: 50.0
    parent_offset {
      x: -0.75
    }
    child_offset {
      x: 0.75
    }
    parent: "HandThumbProximalThree"
    child: "HandThumbMiddleThree"

    rotation { z: -90 }
    angle_limit { min: 0 max: 90 }
    angle_limit { min: -0.1 max: 0.1 }
    angle_limit { min: -0.1 max: 0.1 }
  }
  joints {
    name: "HandPalm_HandThumbProximalThree"
    angular_damping: 50.0
    parent_offset {
      x: -1.5
      z: -.5
    }
    child_offset {
      x: 0.75
    }
    parent: "HandPalm"
    child: "HandThumbProximalThree"

    rotation { z: -90 }
    angle_limit { min: -45 max: 45 }
    angle_limit { min: -45 max: 45 }
    angle_limit { min: -0.1 max: 0.1 }
  }
  joints {
    name: "HandThumbMiddleFour_HandThumbDistalFour"
    angular_damping: 50.0
    parent_offset {
      x: 0.75
    }
    child_offset {
      x: -0.75
    }
    parent: "HandThumbMiddleFour"
    child: "HandThumbDistalFour"

    rotation { z: 90 }
    angle_limit { min: 0 max: 90 }
    angle_limit { min: -0.1 max: 0.1 }
    angle_limit { min: -0.1 max: 0.1 }
  }
  joints {
    name: "HandThumbProximalFour_HandThumbMiddleFour"
    angular_damping: 50.0
    parent_offset {
      x: 0.75
    }
    child_offset {
      x: -0.75
    }
    parent: "HandThumbProximalFour"
    child: "HandThumbMiddleFour"

    rotation { z: 90 }
    angle_limit { min: 0 max: 90 }
    angle_limit { min: -0.1 max: 0.1 }
    angle_limit { min: -0.1 max: 0.1 }
  }
  joints {
    name: "HandPalm_HandThumbProximalFour"
    angular_damping: 50.0
    parent_offset {
      x: 1.5
      z: -.5
    }
    child_offset {
      x: -0.75
    }
    parent: "HandPalm"
    child: "HandThumbProximalFour"

    rotation { x: 180 z: -90 }
    angle_limit { min: -45 max: 45 }
    angle_limit { min: -45 max: 45 }
    angle_limit { min: -0.1 max: 0.1 }
  }
  actuators {
    name: "HandThumbMiddle_HandThumbDistal"
    joint: "HandThumbMiddle_HandThumbDistal"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandThumbProximal_HandThumbMiddle"
    joint: "HandThumbProximal_HandThumbMiddle"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandPalm_HandThumbProximal"
    joint: "HandPalm_HandThumbProximal"
     strength: 300.0
    angle {}
  }
  actuators {
    name: "HandThumbMiddleTwo_HandThumbDistalTwo"
    joint: "HandThumbMiddleTwo_HandThumbDistalTwo"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandThumbProximalTwo_HandThumbMiddleTwo"
    joint: "HandThumbProximalTwo_HandThumbMiddleTwo"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandPalm_HandThumbProximalTwo"
    joint: "HandPalm_HandThumbProximalTwo"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandThumbMiddleThree_HandThumbDistalThree"
    joint: "HandThumbMiddleThree_HandThumbDistalThree"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandThumbProximalThree_HandThumbMiddleThree"
    joint: "HandThumbProximalThree_HandThumbMiddleThree"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandPalm_HandThumbProximalThree"
    joint: "HandPalm_HandThumbProximalThree"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandThumbMiddleFour_HandThumbDistalFour"
    joint: "HandThumbMiddleFour_HandThumbDistalFour"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandThumbProximalFour_HandThumbMiddleFour"
    joint: "HandThumbProximalFour_HandThumbMiddleFour"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandPalm_HandThumbProximalFour"
    joint: "HandPalm_HandThumbProximalFour"
    strength: 300.0
    angle {}
  }
  friction: 1.0
  gravity {
    z: -9.800000190734863
  }
  angular_damping: -0.05000000074505806
  collide_include {
    first: "Ground"
    second: "Object"
  }
  collide_include {
    first: "HandThumbDistal"
    second: "Object"
  }
  collide_include {
    first: "HandThumbDistalTwo"
    second: "Object"
  }
  collide_include {
    first: "HandThumbDistalThree"
    second: "Object"
  }
  collide_include {
    first: "HandThumbDistalFour"
    second: "Object"
  }
  collide_include {
    first: "HandPalm"
    second: "Object"
  }
  dt: 0.02
  substeps: 4
  dynamics_mode: "pbd"
  """

_SYSTEM_CONFIG_SPRING = """
  bodies {
    name: "Ground"
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
  bodies {
    name: "Object"
    colliders {
      capsule {
        radius: 1.0
        length: 2.02
      }
      rotation { x: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbProximal"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { x: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbDistal"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { x: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbMiddle"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { x: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandPalm"
    colliders {
      capsule {
        radius: 1.5
        length: 3.1
      }
      rotation { x: 120 }
    }
    inertia {
      x: 5000.0350000858306885
      y: 10000.7224998474121094
      z: 10000.7224998474121094
    }
    mass: 1.0
    frozen { all : true }
  }
  bodies {
    name: "HandThumbProximalTwo"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { x: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbMiddleTwo"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { x: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbDistalTwo"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { x: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbProximalThree"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { y: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbMiddleThree"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { y: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbDistalThree"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { y: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbProximalFour"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { y: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbMiddleFour"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { y: 90 }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 1.0
  }
  bodies {
    name: "HandThumbDistalFour"
    colliders {
      capsule {
        radius: 0.5
        length: 2.0
      }
      rotation { y: 90 }
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
    colliders { sphere { radius: 1.1 }}
    frozen { all: true }
  }
  joints {
    name: "HandThumbMiddle_HandThumbDistal"
    parent_offset {
      y: 0.75
    }
    child_offset {
      y: -0.75
    }
    parent: "HandThumbMiddle"
    child: "HandThumbDistal"

    stiffness: 5000.0
    angle_limit { min: -90.0 }
    angular_damping: 50.0
  }
  joints {
    name: "HandThumbProximal_HandThumbMiddle"
    parent_offset {
      y: 0.75
    }
    child_offset {
      y: -0.75
    }
    parent: "HandThumbProximal"
    child: "HandThumbMiddle"

    stiffness: 5000.0
    angle_limit { min: -90.0 }
    angular_damping: 50.0
  }
  joints {
    name: "HandPalm_HandThumbProximal"
    parent_offset {
      y: 1.5
      z: -.5
    }
    child_offset {
      y: -0.75
    }
    parent: "HandPalm"
    child: "HandThumbProximal"

    stiffness: 5000.0
    angle_limit { min: -45 max: 45 }
    angle_limit { min: -45 max: 45 }
    angular_damping: 50.0
  }
  joints {
    name: "HandThumbMiddleTwo_HandThumbDistalTwo"
    parent_offset {
      y: -0.75
    }
    child_offset {
      y: 0.75
    }
    parent: "HandThumbMiddleTwo"
    child: "HandThumbDistalTwo"

    stiffness: 5000.0
    angle_limit { min: 0 max: 90 }
    angular_damping: 50.0

  }
  joints {
    name: "HandThumbProximalTwo_HandThumbMiddleTwo"
    parent_offset {
      y: -0.75
    }
    child_offset {
      y: 0.75
    }
    parent: "HandThumbProximalTwo"
    child: "HandThumbMiddleTwo"

    stiffness: 5000.0
    angle_limit { min: 0 max: 90 }
    angular_damping: 50.0
  }
  joints {
    name: "HandPalm_HandThumbProximalTwo"
    parent_offset {
      y: -1.5
      z: -.5
    }
    child_offset {
      y: 0.75
    }
    parent: "HandPalm"
    child: "HandThumbProximalTwo"

    stiffness: 5000.0
    angle_limit { min: -45 max: 45 }
    angle_limit { min: -45 max: 45 }
    angular_damping: 50.0
  }
  joints {
    name: "HandThumbMiddleThree_HandThumbDistalThree"
    parent_offset {
      x: -0.75
    }
    child_offset {
      x: 0.75
    }
    parent: "HandThumbMiddleThree"
    child: "HandThumbDistalThree"

    stiffness: 5000.0
    rotation { z: -90 }
    angle_limit { min: 0 max: 90 }
    angular_damping: 50.0
  }
  joints {
    name: "HandThumbProximalThree_HandThumbMiddleThree"
    parent_offset {
      x: -0.75
    }
    child_offset {
      x: 0.75
    }
    parent: "HandThumbProximalThree"
    child: "HandThumbMiddleThree"

    stiffness: 5000.0
    rotation { z: -90 }
    angle_limit { min: 0 max: 90 }
    angular_damping: 50.0
  }
  joints {
    name: "HandPalm_HandThumbProximalThree"
    parent_offset {
      x: -1.5
      z: -.5
    }
    child_offset {
      x: 0.75
    }
    parent: "HandPalm"
    child: "HandThumbProximalThree"

    stiffness: 5000.0
    rotation { z: -90 }
    angle_limit { min: -45 max: 45 }
    angle_limit { min: -45 max: 45 }
    angular_damping: 50.0
  }
  joints {
    name: "HandThumbMiddleFour_HandThumbDistalFour"
    parent_offset {
      x: 0.75
    }
    child_offset {
      x: -0.75
    }
    parent: "HandThumbMiddleFour"
    child: "HandThumbDistalFour"

    stiffness: 5000.0
    rotation { z: 90 }
    angle_limit { min: 0 max: 90 }
    angular_damping: 50.0
  }
  joints {
    name: "HandThumbProximalFour_HandThumbMiddleFour"
    parent_offset {
      x: 0.75
    }
    child_offset {
      x: -0.75
    }
    parent: "HandThumbProximalFour"
    child: "HandThumbMiddleFour"

    stiffness: 5000.0
    rotation { z: 90 }
    angle_limit { min: 0 max: 90 }
    angular_damping: 50.0
  }
  joints {
    name: "HandPalm_HandThumbProximalFour"
    parent_offset {
      x: 1.5
      z: -.5
    }
    child_offset {
      x: -0.75
    }
    parent: "HandPalm"
    child: "HandThumbProximalFour"

    stiffness: 5000.0
    rotation { x: 180 z: -90 }
    angle_limit { min: -45 max: 45 }
    angle_limit { min: -45 max: 45 }
    angular_damping: 50.0
  }
  actuators {
    name: "HandThumbMiddle_HandThumbDistal"
    joint: "HandThumbMiddle_HandThumbDistal"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandThumbProximal_HandThumbMiddle"
    joint: "HandThumbProximal_HandThumbMiddle"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandPalm_HandThumbProximal"
    joint: "HandPalm_HandThumbProximal"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandThumbMiddleTwo_HandThumbDistalTwo"
    joint: "HandThumbMiddleTwo_HandThumbDistalTwo"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandThumbProximalTwo_HandThumbMiddleTwo"
    joint: "HandThumbProximalTwo_HandThumbMiddleTwo"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandPalm_HandThumbProximalTwo"
    joint: "HandPalm_HandThumbProximalTwo"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandThumbMiddleThree_HandThumbDistalThree"
    joint: "HandThumbMiddleThree_HandThumbDistalThree"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandThumbProximalThree_HandThumbMiddleThree"
    joint: "HandThumbProximalThree_HandThumbMiddleThree"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandPalm_HandThumbProximalThree"
    joint: "HandPalm_HandThumbProximalThree"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandThumbMiddleFour_HandThumbDistalFour"
    joint: "HandThumbMiddleFour_HandThumbDistalFour"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandThumbProximalFour_HandThumbMiddleFour"
    joint: "HandThumbProximalFour_HandThumbMiddleFour"
    strength: 300.0
    angle {}
  }
  actuators {
    name: "HandPalm_HandThumbProximalFour"
    joint: "HandPalm_HandThumbProximalFour"
    strength: 300.0
    angle {}
  }
  friction: 0.77459666924
  gravity {
    z: -9.800000190734863
  }
  angular_damping: -0.05000000074505806
  baumgarte_erp: 0.15000000149011612
  collide_include {
    first: "Ground"
    second: "Object"
  }
  collide_include {
    first: "HandThumbDistal"
    second: "Object"
  }
  collide_include {
    first: "HandThumbDistalTwo"
    second: "Object"
  }
  collide_include {
    first: "HandThumbDistalThree"
    second: "Object"
  }
  collide_include {
    first: "HandThumbDistalFour"
    second: "Object"
  }
  collide_include {
    first: "HandPalm"
    second: "Object"
  }
  dt: 0.02
  substeps: 4
  dynamics_mode: "legacy_spring"
  """
