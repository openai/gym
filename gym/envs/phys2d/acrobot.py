"""An acrobot environment."""

import jumpy as jp

from gym.envs.jax_env import JaxEnv, JaxState


class Acrobot(JaxEnv):
    """Trains an acrobot to swingup and balance.

    Observations:
        0. Theta 0
        1. Theta 1
        2. dTheta 0
        3. dTheta 1
    Actions:
        0. Torque at the elbow joint
    """

    def __init__(self, legacy_spring: bool = False, **kwargs):
        config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
        super().__init__(config=config, **kwargs)

    def internal_reset(self, rng: jp.ndarray) -> JaxState:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jp.random_split(rng, 3)

        qpos = self.sys.default_angle() + jp.random_uniform(
            rng1, (self.sys.num_joint_dof,), -0.01, 0.01
        )
        qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -0.01, 0.01)
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)

        (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)
        obs = self._get_obs(joint_angle, joint_vel)

        reward, terminate, zero = jp.zeros(3)
        metrics = {
            "dist_penalty": zero,
            "vel_penalty": zero,
            "alive_bonus": zero,
            "r_tot": zero,
        }
        return JaxState(qp, obs, reward, terminate, metrics)

    def internal_step(self, state: JaxState, action: jp.ndarray) -> JaxState:
        """Run one timestep of the environment's dynamics."""
        qp, info = self.sys.step(state.qp, action)
        (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)
        obs = self._get_obs(joint_angle, joint_vel)

        alive_bonus = 10.0
        dist_penalty = joint_angle[0] ** 2 + joint_angle[1] ** 2
        vel_penalty = 1e-3 * (joint_vel[0] ** 2 + joint_vel[1] ** 2)
        r = alive_bonus - dist_penalty - vel_penalty
        terminate = jp.zeros(())

        state.metrics.update(
            dist_penalty=dist_penalty, vel_penalty=vel_penalty, r_tot=r
        )

        return state.replace(qp=qp, obs=obs, reward=r, terminate=terminate)

    @property
    def action_size(self):
        return 1

    def _get_obs(self, joint_angle: jp.ndarray, joint_vel: jp.ndarray) -> jp.ndarray:
        """Observe acrobot body position and velocities."""
        return jp.concatenate((joint_angle, joint_vel))


_SYSTEM_CONFIG = """
  bodies {
    name: "cart"
    colliders {
      rotation {
      x: 90
      z: 90
      }
      capsule {
        radius: 0.1
        length: 0.4
      }
    }
    frozen { position { x:1 y:1 z:1 } rotation { x:1 y:1 z:1 } }
    mass: 10.471975
  }
  bodies {
    name: "pole"
    colliders {
      capsule {
        radius: 0.049
        length: 1.0
      }
    }
    frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
    mass: 2.5
  }
  joints {
    name: "hinge"
    parent: "cart"
    child: "pole"
    child_offset { z: -.45 }
    rotation {
      z: 90.0
    }
    angle_limit { min: -360.0 max: 360.0 }
  }
  bodies {
    name: "pole2"
    colliders {
      capsule {
        radius: 0.049
        length:  1.00
      }
    }
    frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
    mass: 2.5
  }
  joints {
    name: "hinge2"
    parent: "pole"
    child: "pole2"
    parent_offset { z: .45 }
    child_offset { z: -.45 }
    rotation {
      z: 90.0
    }
    angle_limit { min: -360.0 max: 360.0 }
  }
  actuators{
    name: "hinge2"
    joint: "hinge2"
    strength: 25.0
    torque{
    }
  }
  defaults {
      angles {
          name: "hinge"
          angle{ x: 180.0 y: 0.0 z: 0.0}
      }
  }
  collide_include {}
  gravity {
    z: -9.81
  }
  dt: 0.01
  substeps: 4
  dynamics_mode: "pbd"
  """

_SYSTEM_CONFIG_SPRING = """
  bodies {
    name: "cart"
    colliders {
      rotation {
      x: 90
      z: 90
      }
      capsule {
        radius: 0.1
        length: 0.4
      }
    }
    frozen { position { x:1 y:1 z:1 } rotation { x:1 y:1 z:1 } }
    mass: 10.471975
  }
  bodies {
    name: "pole"
    colliders {
      capsule {
        radius: 0.049
        length: 1.0
      }
    }
    frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
    mass: 2.5
  }
  joints {
    name: "hinge"
    stiffness: 30000.0
    parent: "cart"
    child: "pole"
    child_offset { z: -.45 }
    rotation {
      z: 90.0
    }
    limit_strength: 0.0
    spring_damping: 500.0
    angle_limit { min: -360.0 max: 360.0 }
  }
  bodies {
    name: "pole2"
    colliders {
      capsule {
        radius: 0.049
        length:  1.00
      }
    }
    frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
    mass: 2.5
  }
  joints {
    name: "hinge2"
    stiffness: 30000.0
    parent: "pole"
    child: "pole2"
    parent_offset { z: .45 }
    child_offset { z: -.45 }
    rotation {
      z: 90.0
    }
    limit_strength: 0.0
    spring_damping: 500.0
    angle_limit { min: -360.0 max: 360.0 }
  }
  actuators{
    name: "hinge2"
    joint: "hinge2"
    strength: 25.0
    torque{
    }
  }
  defaults {
      angles {
          name: "hinge"
          angle{ x: 180.0 y: 0.0 z: 0.0}
      }
  }
  collide_include {}
  gravity {
    z: -9.81
  }
  dt: 0.01
  substeps: 4
  dynamics_mode: "legacy_spring"
  """
