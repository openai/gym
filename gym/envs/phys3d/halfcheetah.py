"""Trains a halfcheetah to run in the +x direction."""

import brax
import jumpy as jp

from gym.envs.phys3d.env import BraxEnv, BraxState


class Halfcheetah(BraxEnv):
    """Trains a halfcheetah to run in the +x direction."""

    def __init__(self, legacy_spring: bool = False, **kwargs):
        config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
        super().__init__(config=config, **kwargs)

    def brax_reset(self, rng: jp.ndarray) -> BraxState:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jp.random_split(rng, 3)
        qpos = self.sys.default_angle() + jp.random_uniform(
            rng1, (self.sys.num_joint_dof,), -0.1, 0.1
        )
        qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -0.1, 0.1)
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        info = self.sys.info(qp)
        obs = self._get_obs(qp, info)
        reward, terminate, zero = jp.zeros(3)
        metrics = {
            "reward_ctrl_cost": zero,
            "reward_forward": zero,
        }
        return BraxState(qp, obs, reward, terminate, metrics)

    def brax_step(self, state: BraxState, action: jp.ndarray) -> BraxState:
        """Run one timestep of the environment's dynamics."""
        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info)

        x_before = state.qp.pos[0, 0]
        x_after = qp.pos[0, 0]
        forward_reward = (x_after - x_before) / self.sys.config.dt
        ctrl_cost = -0.1 * jp.sum(jp.square(action))
        reward = forward_reward + ctrl_cost
        state.metrics.update(reward_ctrl_cost=ctrl_cost, reward_forward=forward_reward)

        return state.replace(qp=qp, obs=obs, reward=reward)

    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        """Observe halfcheetah body position and velocities."""
        # some pre-processing to pull joint angles and velocities
        (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

        # qpos:
        # Z of the torso (1,)
        # orientation of the torso as quaternion (4,)
        # joint angles (8,)
        qpos = [qp.pos[0, 2:], qp.rot[0], joint_angle]

        # qvel:
        # velocity of the torso (3,)
        # angular velocity of the torso (3,)
        # joint angle velocities (8,)
        qvel = [qp.vel[0], qp.ang[0], joint_vel]

        return jp.concatenate(qpos + qvel)


_SYSTEM_CONFIG = """
  bodies {
    name: "torso"
    colliders {
      rotation {
        y: 90.0
      }
      capsule {
        radius: 0.04600000008940697
        length: 1.0920000076293945
      }
    }
    colliders {
      position {
        x: 0.6000000238418579
        z: 0.10000000149011612
      }
      rotation {
        y: 49.847328186035156
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.3919999897480011
      }
    }
    inertia {
      x: 0.9447969794273376
      y: 0.9447969794273376
      z: 0.9447969794273376
    }
    mass: 9.457332611083984
  }
  bodies {
    name: "bthigh"
    colliders {
      position {
        x: 0.10000000149011612
        z: -0.12999999523162842
      }
      rotation {
        x: -180.0
        y: 37.723960876464844
        z: -180.0
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.38199999928474426
      }
    }
    inertia {
      x: 0.029636280611157417
      y: 0.029636280611157417
      z: 0.029636280611157417
    }
    mass: 2.335526943206787
  }
  bodies {
    name: "bshin"
    colliders {
      position {
        x: -0.14000000059604645
        z: -0.07000000029802322
      }
      rotation {
        x: 180.0
        y: -63.68956756591797
        z: 180.0
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.3919999897480011
      }
    }
    inertia {
      x: 0.032029107213020325
      y: 0.032029107213020325
      z: 0.032029107213020325
    }
    mass: 2.402003049850464
  }
  bodies {
    name: "bfoot"
    colliders {
      position {
        x: 0.029999999329447746
        z: -0.09700000286102295
      }
      rotation {
        y: -15.469860076904297
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.2800000011920929
      }
    }
    inertia {
      x: 0.0117056118324399
      y: 0.0117056118324399
      z: 0.0117056118324399
    }
    mass: 1.6574708223342896
  }
  bodies {
    name: "fthigh"
    colliders {
      position {
        x: -0.07000000029802322
        z: -0.11999999731779099
      }
      rotation {
        y: 29.793806076049805
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.3580000102519989
      }
    }
    inertia {
      x: 0.024391336366534233
      y: 0.024391336366534233
      z: 0.024391336366534233
    }
    mass: 2.1759843826293945
  }
  bodies {
    name: "fshin"
    colliders {
      position {
        x: 0.06499999761581421
        z: -0.09000000357627869
      }
      rotation {
        y: -34.37746810913086
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.30399999022483826
      }
    }
    inertia {
      x: 0.014954624697566032
      y: 0.014954624697566032
      z: 0.014954624697566032
    }
    mass: 1.8170133829116821
  }
  bodies {
    name: "ffoot"
    colliders {
      position {
        x: 0.04500000178813934
        z: -0.07000000029802322
      }
      rotation {
        y: -34.37746810913086
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.23199999332427979
      }
    }
    inertia {
      x: 0.006711110472679138
      y: 0.006711110472679138
      z: 0.006711110472679138
    }
    mass: 1.3383854627609253
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
    frozen {
      position { x: 1.0 y: 1.0 z: 1.0 }
      rotation { x: 1.0 y: 1.0 z: 1.0 }
    }
  }
  joints {
    name: "bthigh"
    parent: "torso"
    child: "bthigh"
    parent_offset {
      x: -0.5
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angle_limit {
      min: -29.793806076049805
      max: 60.16056823730469
    }
    }
  joints {
    name: "bshin"
    parent: "bthigh"
    child: "bshin"
    parent_offset {
      x: 0.1599999964237213
      z: -0.25
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angle_limit {
      min: -44.97718811035156
      max: 44.97718811035156
    }
    }
  joints {
    name: "bfoot"
    parent: "bshin"
    child: "bfoot"
    parent_offset {
      x: -0.2800000011920929
      z: -0.14000000059604645
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angle_limit {
      min: -22.918312072753906
      max: 44.97718811035156
    }
    }
  joints {
    name: "fthigh"
    parent: "torso"
    child: "fthigh"
    parent_offset {
      x: 0.5
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angle_limit {
      min: -57.295780181884766
      max: 40.1070442199707
    }
    }
  joints {
    name: "fshin"
    parent: "fthigh"
    child: "fshin"
    parent_offset {
      x: -0.14000000059604645
      z: -0.23999999463558197
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angle_limit {
      min: -68.75493621826172
      max: 49.847328186035156
    }
    }
  joints {
    name: "ffoot"
    parent: "fshin"
    child: "ffoot"
    parent_offset {
      x: 0.12999999523162842
      z: -0.18000000715255737
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    angle_limit {
      min: -28.647890090942383
      max: 28.647890090942383
    }
    }
  actuators {
    name: "bthigh"
    joint: "bthigh"
    strength: 120.0
    torque {
    }
  }
  actuators {
    name: "bshin"
    joint: "bshin"
    strength: 90.0
    torque {
    }
  }
  actuators {
    name: "bfoot"
    joint: "bfoot"
    strength: 60
    torque {
    }
  }
  actuators {
    name: "fthigh"
    joint: "fthigh"
    strength: 120.0
    torque {
    }
  }
  actuators {
    name: "fshin"
    joint: "fshin"
    strength: 60
    torque {
    }
  }
  actuators {
    name: "ffoot"
    joint: "ffoot"
    strength: 30.0
    torque {
    }
  }
  friction: 0.77459666924
  gravity {
    z: -9.8100004196167
  }
  angular_damping: -0.009999999776482582
  collide_include {
    first: "floor"
    second: "torso"
  }
  collide_include {
    first: "floor"
    second: "bfoot"
  }
  collide_include {
    first: "floor"
    second: "ffoot"
  }
  collide_include {
    first: "floor"
    second: "bthigh"
  }
  collide_include {
    first: "floor"
    second: "fthigh"
  }
  collide_include {
    first: "floor"
    second: "bshin"
  }
  collide_include {
    first: "floor"
    second: "fshin"
  }
  collide_include {
    first: "bfoot"
    second: "ffoot"
  }
  dt: 0.05
  substeps: 16
  frozen {
    position {
      y: 1.0
    }
    rotation {
      x: 1.0
      z: 1.0
    }
  }
  dynamics_mode: "pbd"
"""

_SYSTEM_CONFIG_SPRING = """
  bodies {
    name: "torso"
    colliders {
      rotation {
        y: 90.0
      }
      capsule {
        radius: 0.04600000008940697
        length: 1.0920000076293945
      }
    }
    colliders {
      position {
        x: 0.6000000238418579
        z: 0.10000000149011612
      }
      rotation {
        y: 49.847328186035156
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.3919999897480011
      }
    }
    inertia {
      x: 0.9447969794273376
      y: 0.9447969794273376
      z: 0.9447969794273376
    }
    mass: 9.457332611083984
  }
  bodies {
    name: "bthigh"
    colliders {
      position {
        x: 0.10000000149011612
        z: -0.12999999523162842
      }
      rotation {
        x: -180.0
        y: 37.723960876464844
        z: -180.0
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.38199999928474426
      }
    }
    inertia {
      x: 0.029636280611157417
      y: 0.029636280611157417
      z: 0.029636280611157417
    }
    mass: 2.335526943206787
  }
  bodies {
    name: "bshin"
    colliders {
      position {
        x: -0.14000000059604645
        z: -0.07000000029802322
      }
      rotation {
        x: 180.0
        y: -63.68956756591797
        z: 180.0
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.3919999897480011
      }
    }
    inertia {
      x: 0.032029107213020325
      y: 0.032029107213020325
      z: 0.032029107213020325
    }
    mass: 2.402003049850464
  }
  bodies {
    name: "bfoot"
    colliders {
      position {
        x: 0.029999999329447746
        z: -0.09700000286102295
      }
      rotation {
        y: -15.469860076904297
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.2800000011920929
      }
    }
    inertia {
      x: 0.0117056118324399
      y: 0.0117056118324399
      z: 0.0117056118324399
    }
    mass: 1.6574708223342896
  }
  bodies {
    name: "fthigh"
    colliders {
      position {
        x: -0.07000000029802322
        z: -0.11999999731779099
      }
      rotation {
        y: 29.793806076049805
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.3580000102519989
      }
    }
    inertia {
      x: 0.024391336366534233
      y: 0.024391336366534233
      z: 0.024391336366534233
    }
    mass: 2.1759843826293945
  }
  bodies {
    name: "fshin"
    colliders {
      position {
        x: 0.06499999761581421
        z: -0.09000000357627869
      }
      rotation {
        y: -34.37746810913086
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.30399999022483826
      }
    }
    inertia {
      x: 0.014954624697566032
      y: 0.014954624697566032
      z: 0.014954624697566032
    }
    mass: 1.8170133829116821
  }
  bodies {
    name: "ffoot"
    colliders {
      position {
        x: 0.04500000178813934
        z: -0.07000000029802322
      }
      rotation {
        y: -34.37746810913086
      }
      capsule {
        radius: 0.04600000008940697
        length: 0.23199999332427979
      }
    }
    inertia {
      x: 0.006711110472679138
      y: 0.006711110472679138
      z: 0.006711110472679138
    }
    mass: 1.3383854627609253
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
    frozen {
      position { x: 1.0 y: 1.0 z: 1.0 }
      rotation { x: 1.0 y: 1.0 z: 1.0 }
    }
  }
  joints {
    name: "bthigh"
    stiffness: 25000
    parent: "torso"
    child: "bthigh"
    parent_offset {
      x: -0.5
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    spring_damping: 60
    angle_limit {
      min: -29.793806076049805
      max: 60.16056823730469
    }
    limit_strength: 1000
    }
  joints {
    name: "bshin"
    stiffness: 25000
    parent: "bthigh"
    child: "bshin"
    parent_offset {
      x: 0.1599999964237213
      z: -0.25
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    spring_damping: 60
    angle_limit {
      min: -44.97718811035156
      max: 44.97718811035156
    }
    limit_strength: 1000
    }
  joints {
    name: "bfoot"
    stiffness: 25000
    parent: "bshin"
    child: "bfoot"
    parent_offset {
      x: -0.2800000011920929
      z: -0.14000000059604645
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    spring_damping: 60
    angle_limit {
      min: -22.918312072753906
      max: 44.97718811035156
    }
    limit_strength: 1000
    }
  joints {
    name: "fthigh"
    stiffness: 25000
    parent: "torso"
    child: "fthigh"
    parent_offset {
      x: 0.5
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    spring_damping: 60
    angle_limit {
      min: -57.295780181884766
      max: 40.1070442199707
    }
    limit_strength: 1000
    }
  joints {
    name: "fshin"
    stiffness: 25000
    parent: "fthigh"
    child: "fshin"
    parent_offset {
      x: -0.14000000059604645
      z: -0.23999999463558197
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    spring_damping: 80.0
    angle_limit {
      min: -68.75493621826172
      max: 49.847328186035156
    }
    limit_strength: 1000
    }
  joints {
    name: "ffoot"
    stiffness: 25000
    parent: "fshin"
    child: "ffoot"
    parent_offset {
      x: 0.12999999523162842
      z: -0.18000000715255737
    }
    child_offset {
    }
    rotation {
      z: 90.0
    }
    spring_damping: 50.0
    angle_limit {
      min: -28.647890090942383
      max: 28.647890090942383
    }
    limit_strength: 1000
    }
  actuators {
    name: "bthigh"
    joint: "bthigh"
    strength: 120.0
    torque {
    }
  }
  actuators {
    name: "bshin"
    joint: "bshin"
    strength: 90.0
    torque {
    }
  }
  actuators {
    name: "bfoot"
    joint: "bfoot"
    strength: 60
    torque {
    }
  }
  actuators {
    name: "fthigh"
    joint: "fthigh"
    strength: 120.0
    torque {
    }
  }
  actuators {
    name: "fshin"
    joint: "fshin"
    strength: 60
    torque {
    }
  }
  actuators {
    name: "ffoot"
    joint: "ffoot"
    strength: 30.0
    torque {
    }
  }
  friction: 0.77459666924
  gravity {
    z: -9.8100004196167
  }
  angular_damping: -0.009999999776482582
  baumgarte_erp: 0.20000000149011612
  collide_include {
    first: "floor"
    second: "torso"
  }
  collide_include {
    first: "floor"
    second: "bfoot"
  }
  collide_include {
    first: "floor"
    second: "ffoot"
  }
  collide_include {
    first: "floor"
    second: "bthigh"
  }
  collide_include {
    first: "floor"
    second: "fthigh"
  }
  collide_include {
    first: "floor"
    second: "bshin"
  }
  collide_include {
    first: "floor"
    second: "fshin"
  }
  collide_include {
    first: "bfoot"
    second: "ffoot"
  }
  dt: 0.05
  substeps: 16
  frozen {
    position {
      y: 1.0
    }
    rotation {
      x: 1.0
      z: 1.0
    }
  }
  dynamics_mode: "legacy_spring"
  """
