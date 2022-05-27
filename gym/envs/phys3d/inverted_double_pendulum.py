"""An inverted pendulum environment."""

import brax
import jumpy as jp

from gym.envs.phys3d.env import BraxEnv, BraxState


class InvertedDoublePendulum(BraxEnv):
    """Trains an inverted pendulum to remain stationary."""

    def __init__(self, legacy_spring: bool = False, **kwargs):
        config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
        super().__init__(config=config, **kwargs)

    def brax_reset(self, rng: jp.ndarray) -> BraxState:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jp.random_split(rng, 3)
        qpos = self.sys.default_angle() + jp.random_uniform(
            rng1, (self.sys.num_joint_dof,), -0.01, 0.01
        )
        qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -0.01, 0.01)
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        info = self.sys.info(qp)
        (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)
        obs = self._get_obs(qp, info, joint_angle, joint_vel)
        reward, terminate, zero = jp.zeros(3)
        metrics = {
            "dist_penalty": zero,
            "vel_penalty": zero,
            "alive_bonus": zero,
            "r_tot": zero,
        }
        return BraxState(qp, obs, reward, terminate, metrics)

    def brax_step(self, state: BraxState, action: jp.ndarray) -> BraxState:
        """Run one timestep of the environment's dynamics."""
        qp, info = self.sys.step(state.qp, action)
        (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)
        obs = self._get_obs(qp, info, joint_angle, joint_vel)
        tip_pos = jp.take(state.qp, 2).to_world(jp.array([0, 0, 0.3]))
        (x, _, y), _ = tip_pos
        dist_penalty = 0.01 * x**2 + (y - 2) ** 2
        v1, v2 = joint_vel
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10.0
        r = alive_bonus - dist_penalty - vel_penalty
        terminate = jp.where(y <= 1, jp.float32(1), jp.float32(0))
        state.metrics.update(
            dist_penalty=dist_penalty,
            vel_penalty=vel_penalty,
            alive_bonus=alive_bonus,
            r_tot=r,
        )

        return state.replace(qp=qp, obs=obs, reward=r, terminate=terminate)

    @property
    def action_size(self):
        return 1

    def _get_obs(
        self,
        qp: brax.QP,
        info: brax.Info,
        joint_angle: jp.ndarray,
        joint_vel: jp.ndarray,
    ) -> jp.ndarray:
        """Observe cartpole body position and velocities."""

        position_obs = [
            jp.array([qp.pos[0, 0]]),  # cart x pos
            jp.sin(joint_angle),
            jp.cos(joint_angle),
        ]

        # qvel:
        qvel = [jp.array([qp.vel[0, 0]]), joint_vel]  # cart x vel

        return jp.concatenate(position_obs + qvel)


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
    frozen { position { x:0 y:1 z:1 } rotation { x:1 y:1 z:1 } }
    mass: 10.471975
  }
  bodies {
    name: "pole"
    colliders {
      capsule {
        radius: 0.049
        length: 0.69800085
      }
    }
    frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
    mass: 5.0185914
  }
  joints {
    name: "hinge"
    parent: "cart"
    child: "pole"
    child_offset { z: -.3 }
    rotation {
      z: 90.0
    }
    angle_limit { min: 0.0 max: 0.0 }
  }
  bodies {
    name: "pole2"
    colliders {
      capsule {
        radius: 0.049
        length: 0.69800085
      }
    }
    frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
    mass: 5.0185914
  }
  joints {
    name: "hinge2"
    parent: "pole"
    child: "pole2"
    parent_offset { z: .3 }
    child_offset { z: -.3 }
    rotation {
      z: 90.0
    }
    angle_limit { min: 0.0 max: 0.0 }
  }
  forces {
    name: "cart_thruster"
    body: "cart"
    strength: 500.0
    thruster {}
  }
  collide_include {}
  gravity {
    z: -9.81
  }
  dt: 0.05
  substeps: 12
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
    frozen { position { x:0 y:1 z:1 } rotation { x:1 y:1 z:1 } }
    mass: 10.471975
  }
  bodies {
    name: "pole"
    colliders {
      capsule {
        radius: 0.049
        length: 0.69800085
      }
    }
    frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
    mass: 5.0185914
  }
  joints {
    name: "hinge"
    stiffness: 30000.0
    parent: "cart"
    child: "pole"
    child_offset { z: -.3 }
    rotation {
      z: 90.0
    }
    limit_strength: 0.0
    spring_damping: 500.0
    angle_limit { min: 0.0 max: 0.0 }
  }
  bodies {
    name: "pole2"
    colliders {
      capsule {
        radius: 0.049
        length: 0.69800085
      }
    }
    frozen { position { x: 0 y: 1 z: 0 } rotation { x: 1 y: 0 z: 1 } }
    mass: 5.0185914
  }
  joints {
    name: "hinge2"
    stiffness: 30000.0
    parent: "pole"
    child: "pole2"
    parent_offset { z: .3 }
    child_offset { z: -.3 }
    rotation {
      z: 90.0
    }
    limit_strength: 0.0
    spring_damping: 500.0
    angle_limit { min: 0.0 max: 0.0 }
  }
  forces {
    name: "cart_thruster"
    body: "cart"
    strength: 500.0
    thruster {}
  }
  collide_include {}
  gravity {
    z: -9.81
  }
  dt: 0.05
  substeps: 12
  dynamics_mode: "legacy_spring"
  """
