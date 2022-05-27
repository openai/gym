"""An inverted pendulum environment."""

import brax
import jumpy as jp

from gym.envs.phys3d.env import BraxEnv, BraxState


class InvertedPendulum(BraxEnv):
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
        obs = self._get_obs(qp, info)
        reward, terminate, zero = jp.zeros(3)
        metrics = {
            "survive_reward": zero,
        }
        return BraxState(qp, obs, reward, terminate, metrics)

    def brax_step(self, state: BraxState, action: jp.ndarray) -> BraxState:
        """Run one timestep of the environment's dynamics."""
        reward = 1.0
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info)
        terminate = jp.where(qp.pos[1, 2] > 0.2, jp.float32(0), jp.float32(1))
        state.metrics.update(survive_reward=reward)

        return state.replace(qp=qp, obs=obs, reward=reward, terminate=terminate)

    @property
    def action_size(self):
        return 1

    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        """Observe cartpole body position and velocities."""
        # some pre-processing to pull joint angles and velocities
        (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

        # [cart pos, angle, cart vel, angle vel]
        obs = jp.concatenate([qp.pos[0, :1], joint_angle, qp.vel[0, :1], joint_vel])
        return obs


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
  forces {
    name: "cart_thruster"
    body: "cart"
    strength: 100.0
    thruster {}
  }
  collide_include {}
  gravity {
    z: -9.81
  }
  dt: 0.04
  substeps: 8
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
    stiffness: 10000.0
    parent: "cart"
    child: "pole"
    child_offset { z: -.3 }
    rotation {
      z: 90.0
    }
    limit_strength: 0.0
    angle_limit { min: 0.0 max: 0.0 }
  }
  forces {
    name: "cart_thruster"
    body: "cart"
    strength: 100.0
    thruster {}
  }
  collide_include {}
  gravity {
    z: -9.81
  }
  dt: 0.04
  substeps: 8
  dynamics_mode: "legacy_spring"
  """
