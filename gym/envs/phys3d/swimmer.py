"""Trains a swimmer to swim in the +x direction.

Based on the OpenAI Gym MuJoCo Swimmer environment.
"""

import brax
import jumpy as jp
from brax import math

from gym.envs.phys3d.env import BraxEnv, BraxState


class Swimmer(BraxEnv):
    """Trains a swimmer to swim forward."""

    def __init__(self, legacy_spring: bool = False, **kwargs):
        config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
        super().__init__(config=config, **kwargs)

        # these parameters were derived from the mujoco swimmer:
        viscosity = 0.1
        density = 10.0
        inertia = (
            0.17278759594743870,
            3.5709436495803999,
            3.5709436495803999,
        )
        body_mass = 34.557519189487735

        # convert inertia to box
        inertia = jp.array(
            [
                inertia[1] + inertia[2] - inertia[0],
                inertia[0] + inertia[1] - inertia[2],
                inertia[0] + inertia[2] - inertia[1],
            ]
        )
        inertia = jp.sqrt(inertia / (body_mass * 6))

        # spherical drag
        self._spherical_drag = -3 * jp.pi * jp.mean(inertia) * viscosity

        # corrections to spherical drag force due to shape of capsules
        self._fix_drag = (
            0.5
            * density
            * jp.array(
                [
                    inertia[1] * inertia[2],
                    inertia[0] * inertia[2],
                    inertia[0] * inertia[1],
                ]
            )
        )

    def brax_reset(self, rng: jp.ndarray) -> BraxState:
        rng, rng1, rng2 = jp.random_split(rng, 3)
        qpos = self.sys.default_angle() + jp.random_uniform(
            rng1, (self.sys.num_joint_dof,), -0.1, 0.1
        )
        qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -0.005, 0.005)
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        info = self.sys.info(qp)
        obs = self._get_obs(qp, info)
        reward, terminate, zero = jp.zeros(3)
        metrics = {
            "rewardFwd": zero,
            "rewardCtrl": zero,
        }
        return BraxState(qp, obs, reward, terminate, metrics)

    def brax_step(self, state: BraxState, action: jp.ndarray) -> BraxState:
        force = self._get_viscous_force(state.qp)
        act = jp.concatenate([action, force.reshape(-1)], axis=0)
        qp, info = self.sys.step(state.qp, act)
        obs = self._get_obs(qp, info)

        x_before = self._get_center_of_mass(state.qp)[0]
        x_after = self._get_center_of_mass(qp)[0]

        reward_fwd = (x_after - x_before) / self.sys.config.dt
        reward_ctrl = 0.0001 * -jp.square(action).sum()
        reward = reward_fwd + reward_ctrl

        state.metrics.update(
            rewardFwd=reward_fwd,
            rewardCtrl=reward_ctrl,
        )

        return state.replace(qp=qp, obs=obs, reward=reward)

    @property
    def action_size(self):
        return 2

    def _get_viscous_force(self, qp):
        """Calculate viscous force to apply to each body."""
        # ignore the floor
        qp = jp.take(qp, jp.arange(1, qp.vel.shape[0]))

        # spherical drag force
        force = qp.vel * self._spherical_drag

        # corrections to spherical drag force due to shape of capsules
        vel = jp.vmap(math.rotate)(qp.vel, math.quat_inv(qp.rot))
        force -= jp.diag(self._fix_drag * jp.abs(vel) * vel)
        force = jp.vmap(math.rotate)(force, qp.rot)
        force = jp.clip(force, -5.0, 5.0)

        return force

    def _get_center_of_mass(self, qp):
        mass = self.sys.body.mass[1:]
        return jp.sum(jp.vmap(jp.multiply)(mass, qp.pos[1:]), axis=0) / jp.sum(mass)

    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        """Observe swimmer body position and velocities."""
        # some pre-processing to pull joint angles and velocities
        (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

        com = self._get_center_of_mass(qp)
        rel_pos = qp.pos[1:] - com

        qpos = [rel_pos.ravel(), qp.rot.ravel(), joint_angle]
        qvel = [qp.vel.ravel(), qp.ang.ravel(), joint_vel]

        return jp.concatenate(qpos + qvel)


_SYSTEM_CONFIG = """
  bodies {
    name: "floor"
    colliders {
      plane {}
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  bodies {
    name: "torso"
    colliders {
      rotation {
        y: -90.0
      }
      capsule {
        radius: 0.1
        length: 1.2
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.35604717
  }
  bodies {
    name: "mid"
    colliders {
      rotation {
        y: -90.0
      }
      capsule {
        radius: 0.1
        length: 1.2
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.35604717
  }
  bodies {
    name: "back"
    colliders {
      rotation {
        y: -90.0
      }
      capsule {
        radius: 0.1
        length: 1.2
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.35604717
  }
  joints {
    name: "rot2"
    parent: "torso"
    child: "mid"
    parent_offset {
      x: 0.5
    }
    child_offset {
      x: -.5
    }
    rotation {
      y: -90.0
    }
    angle_limit {
      min: -100.0
      max: 100.0
    }
    angular_damping: 10.
    reference_rotation {
    }
  }
  joints {
    name: "rot3"
    parent: "mid"
    child: "back"
    parent_offset {
      x: .5
    }
    child_offset {
      x: -.5
    }
    rotation {
      y: -90.0
    }
    angle_limit {
      min: -100.0
      max: 100.0
    }
    angular_damping: 10.
    reference_rotation {
    }
  }
  actuators {
    name: "rot2"
    joint: "rot2"
    strength: 30.0
    torque {
    }
  }
  actuators {
    name: "rot3"
    joint: "rot3"
    strength: 30.0
    torque {
    }
  }
  forces {
    name: "torso_viscosity_thruster"
    body: "torso"
    strength: 1.0
    thruster {}
  }
  forces {
    name: "mid_viscosity_thruster"
    body: "mid"
    strength: 1.0
    thruster {}
  }
  forces {
    name: "back_viscosity_thruster"
    body: "back"
    strength: 1.0
    thruster {}
  }
  frozen {
    position { z: 0.0 }
    rotation { x: 1.0 y: 1.0 }
  }
  friction: 0.6
  angular_damping: -0.05
  collide_include { }
  dt: 0.02
  substeps: 12
  dynamics_mode: "pbd"
  """

_SYSTEM_CONFIG_SPRING = """
  bodies {
    name: "floor"
    colliders {
      plane {}
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  bodies {
    name: "torso"
    colliders {
      rotation {
        y: -90.0
      }
      capsule {
        radius: 0.1
        length: 1.2
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.35604717
  }
  bodies {
    name: "mid"
    colliders {
      rotation {
        y: -90.0
      }
      capsule {
        radius: 0.1
        length: 1.2
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.35604717
  }
  bodies {
    name: "back"
    colliders {
      rotation {
        y: -90.0
      }
      capsule {
        radius: 0.1
        length: 1.2
      }
    }
    inertia {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    mass: 0.35604717
  }
  joints {
    name: "rot2"
    stiffness: 10000.0
    parent: "torso"
    child: "mid"
    parent_offset {
      x: 0.5
    }
    child_offset {
      x: -.5
    }
    rotation {
      y: -90.0
    }
    angle_limit {
      min: -100.0
      max: 100.0
    }
    angular_damping: 10.
    reference_rotation {
    }
  }
  joints {
    name: "rot3"
    stiffness: 10000.0
    parent: "mid"
    child: "back"
    parent_offset {
      x: .5
    }
    child_offset {
      x: -.5
    }
    rotation {
      y: -90.0
    }
    angle_limit {
      min: -100.0
      max: 100.0
    }
    angular_damping: 10.
    reference_rotation {
    }
  }
  actuators {
    name: "rot2"
    joint: "rot2"
    strength: 30.0
    torque {
    }
  }
  actuators {
    name: "rot3"
    joint: "rot3"
    strength: 30.0
    torque {
    }
  }
  forces {
    name: "torso_viscosity_thruster"
    body: "torso"
    strength: 1.0
    thruster {}
  }
  forces {
    name: "mid_viscosity_thruster"
    body: "mid"
    strength: 1.0
    thruster {}
  }
  forces {
    name: "back_viscosity_thruster"
    body: "back"
    strength: 1.0
    thruster {}
  }
  frozen {
    position { z: 0.0 }
    rotation { x: 1.0 y: 1.0 }
  }
  friction: 0.6
  angular_damping: -0.05
  baumgarte_erp: 0.1
  collide_include { }
  dt: 0.02
  substeps: 12
  dynamics_mode: "legacy_spring"
  """
