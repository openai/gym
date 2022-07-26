from typing import Optional

import brax
import jax
import numpy as np
from brax import jumpy as jp
from google.protobuf import text_format

from brax.io import image

import gym
from gym import spaces
from gym.error import DependencyNotInstalled
from gym.utils import EzPickle
from gym.utils.renderer import Renderer

# TODO: stable physics / working heuristic
# TODO: jit env.step?
# TODO: hardcore mode
# TODO: normalized rewards
# TODO: rendering cosmetics

VIEWPORT_W = 600
VIEWPORT_H = 400

SCALE = 15  # pixels per meter

GROUND, HULL, LEFT_THIGH, RIGHT_THIGH, LEFT_LEG, RIGHT_LEG = range(6)
LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE = range(4)

HULL_MASS = 42.0
LIMB_MASS = 1.5

HULL_X = 1.2
HULL_Z = 0.4

HIP_OFFSET_X = 0.1
THIGH_X = 0.3
THIGH_Z = 1.2

LEG_X = THIGH_X * 0.8
LEG_Z = THIGH_Z

STRENGTH_HIP = 50
STRENGTH_KNEE = STRENGTH_HIP * 4 / 6

TERRAIN_STARTPAD = 60  # in steps
TERRAIN_LENGTH = 240  # in steps
TERRAIN_SCALE = 10
TERRAIN_SIZE = TERRAIN_LENGTH / 2
INITIAL_X = 40 / TERRAIN_LENGTH * TERRAIN_SIZE

LIDAR_RANGE = 10.0


class BipedalWalker(gym.Env, EzPickle):
    """
    ### Description
    This is a simple 4-joint walker robot environment.
    There are two versions:
    - Normal, with slightly uneven terrain.
    - Hardcore, with ladders, stumps, pitfalls.

    To solve the normal version, you need to get 300 points in 1600 time steps.
    To solve the hardcore version, you need 300 points in 2000 time steps.

    A heuristic is provided for testing. It's also useful to get demonstrations
    to learn from. To run the heuristic:
    ```
    python gym/envs/box2d/bipedal_walker.py
    ```

    ### Action Space
    Actions are motor speed values in the [-1, 1] range for each of the
    4 joints at both hips and knees.

    ### Observation Space
    State consists of hull angle speed, angular velocity, horizontal speed,
    vertical speed, position of joints and joints angular speed, legs contact
    with ground, and 10 lidar rangefinder measurements. There are no coordinates
    in the state vector.

    ### Rewards
    Reward is given for moving forward, totaling 300+ points up to the far end.
    If the robot falls, it gets -100. Applying motor torque costs a small
    amount of points. A more optimal agent will get a better score.

    ### Starting State
    The walker starts standing at the left end of the terrain with the hull
    horizontal, and both legs in the same position with a slight knee angle.

    ### Episode Termination
    The episode will terminate if the hull gets in contact with the ground or
    if the walker exceeds the right end of the terrain length.

    ### Arguments
    To use to the _hardcore_ environment, you need to specify the
    `hardcore=True` argument like below:
    ```python
    import gym
    env = gym.make("BipedalWalker-v4", hardcore=True)
    ```

    ### Version History
    - v4: Replaced box2d with brax
    - v3: returns closest lidar trace instead of furthest;
        faster video recording
    - v2: Count energy spent
    - v1: Legs now report contact with ground; motors have higher torque and
        speed; ground has higher friction; lidar rendered less nervously.
    - v0: Initial version


    <!-- ### References -->

    ### Credits
    Created by Oleg Klimov

    """

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 100,
    }

    def __init__(self, render_mode: Optional[str] = None, hardcore: bool = False):
        EzPickle.__init__(self)

        self.hardcore = hardcore

        # we use 5.0 to represent the joints moving at maximum
        # 5 x the rated speed due to impulses from ground contact etc.
        low = np.array(
            [
                -jp.pi,
                -5.0,
                -5.0,
                -5.0,
                -jp.pi,
                -5.0,
                -jp.pi,
                -5.0,
                -0.0,
                -jp.pi,
                -5.0,
                -jp.pi,
                -5.0,
                -0.0,
            ]
            + [-1.0] * 10
        ).astype(np.float32)
        high = np.array(
            [
                jp.pi,
                5.0,
                5.0,
                5.0,
                jp.pi,
                5.0,
                jp.pi,
                5.0,
                5.0,
                jp.pi,
                5.0,
                jp.pi,
                5.0,
                5.0,
            ]
            + [1.0] * 10
        ).astype(np.float32)
        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -1]).astype(np.float32),
            np.array([1, 1, 1, 1]).astype(np.float32),
        )
        self.observation_space = spaces.Box(low, high)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)
        self.screen = None
        self.clock = None

    def _to_2d(self, arr):
        return arr[jp.array([0, 2])]

    def _get_angle(self, rot):
        # returns angle around y axis from rotation quaternion
        return 2 * jp.arctan2(rot[2], rot[0])

    def _get_2d_box_size(self, body):
        body_halfsizes = body.colliders[0].box.halfsize
        return 2 * jp.array([body_halfsizes.x, body_halfsizes.z])

    def _get_lidar(self):
        hull_pos = self._to_2d(self.qp.pos[HULL])
        directions = jp.array(
            [[jp.sin(1.5 * i / 10.0), -jp.cos(1.5 * i / 10.0)] for i in range(10)]
        )
        p1 = self.terrain_points[:-1]
        p2 = self.terrain_points[1:]
        v1 = hull_pos - p1
        v2 = p2 - p1
        v3 = jp.array([-directions.T[1], directions.T[0]]).T
        distances = jp.cross(v2, v1)[:, None] / (v2 @ v3.T)
        intersect_points = hull_pos + np.einsum("ij,ki->ikj", directions, distances)
        valid_intersects = jp.array(
            [
                (
                    (self.terrain_points[:-1, 0] <= intersect_points[i][:, 0])
                    & (intersect_points[i][:, 0] <= self.terrain_points[1:, 0])
                )
                for i in range(len(directions))
            ]
        ).T
        min_distances = jp.minimum(
            LIDAR_RANGE, jp.where(valid_intersects, distances, LIDAR_RANGE)
        ).min(axis=0)
        normed_min_distances = min_distances / LIDAR_RANGE
        return normed_min_distances

    def _get_state(self, info):
        left_leg_ground_contact, right_leg_ground_contact = (
            info.contact.vel[(LEFT_LEG, RIGHT_LEG), 2] != 0
        ).astype(int)
        joint_angle, joint_vel = self.sys.joints[0].angle_vel(self.qp)
        joint_angle += [0, 1, 0, 1]
        joint_vel /= jp.array([6, 4, 6, 4])

        state = [
            self._get_angle(self.qp.rot[HULL]),
            self.qp.ang[HULL][1] / 200,
            *self._to_2d(0.3 * self.qp.vel[HULL]),
            joint_angle[LEFT_HIP],
            joint_vel[LEFT_HIP],
            joint_angle[LEFT_KNEE],
            joint_vel[LEFT_KNEE],
            left_leg_ground_contact,
            joint_angle[RIGHT_HIP],
            joint_vel[RIGHT_HIP],
            joint_angle[RIGHT_KNEE],
            joint_vel[RIGHT_KNEE],
            right_leg_ground_contact,
            *self._get_lidar(),
        ]
        assert len(state) == 24
        return state

    def _generate_terrain(self, hardcore):
        self.terrain_x = np.linspace(
            -INITIAL_X, TERRAIN_SIZE - INITIAL_X, TERRAIN_LENGTH
        )
        velocity = 0
        y = 0
        self.terrain_y = [0] * TERRAIN_STARTPAD

        for _ in range(TERRAIN_LENGTH - TERRAIN_STARTPAD):
            velocity += (
                -0.2 * velocity
                - 0.01 * np.sign(y)
                + self.np_random.uniform(-1, 1) / TERRAIN_SCALE
            )
            y += velocity
            self.terrain_y.append(y)

        self.terrain_y = jp.array(self.terrain_y)
        self.terrain_points = jp.array([self.terrain_x, self.terrain_y]).T
        terrain_map = jp.repeat(self.terrain_y, TERRAIN_LENGTH).tolist()
        return terrain_map

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.renderer.reset()
        self.prev_shaping = None

        terrain_map = self._generate_terrain(self.hardcore)

        _SYSTEM_CONFIG = """
            dt: 0.02
            substeps: 20
            friction: 1.0
            dynamics_mode: "pbd"
            gravity { z: -10 }
            velocity_damping: 1.0
            bodies {
                name: "ground"
                colliders {
                heightMap {
                    size: %s
                    data: %s
                }
                }
                frozen { all: true }
            }
            bodies {
                name: "hull"
                colliders { box { halfsize { x: %s y: 0.5 z: %s }}}
                inertia { x: 1.0 y: 1000.0 z: 1.0 }
                mass: %s
            }
            bodies {
                name: "left_thigh"
                colliders {
                box {
                    halfsize { x: %s y: 0.2 z: %s }
                }
                }
                inertia { x: 1.0 y: 1.0 z: 1.0 }
                mass: %s
            }
            bodies {
                name: "right_thigh"
                colliders {
                box {
                    halfsize { x: %s y: 0.2 z: %s }
                }
                }
                inertia { x: 1.0 y: 1.0 z: 1.0 }
                mass: %s
            }
            bodies {
                name: "left_leg"
                colliders {
                box {
                    halfsize { x: %s y: 0.2 z: %s }
                }
                }
                inertia { x: 1.0 y: 1.0 z: 1.0 }
                mass: %s
            }
            bodies {
                name: "right_leg"
                colliders {
                box {
                    halfsize { x: %s y: 0.2 z: %s }
                }
                }
                inertia { x: 1.0 y: 1.0 z: 1.0 }
                mass: %s
            }
            joints {
                name: "left_hip"
                parent: "hull"
                child: "left_thigh"
                parent_offset { x: %s z: -%s }
                child_offset { x: %s z: %s }
                rotation { z: -90.0 y: 0.0 }
                angle_limit { min: -46.0 max: 63.0 }
                angular_damping: 1.0
            }
            joints {
                name: "right_hip"
                parent: "hull"
                child: "right_thigh"
                parent_offset { x: %s z: -%s }
                child_offset { x: %s z: %s }
                rotation { z: -90.0 y: 0.0 }
                angle_limit { min: -46.0 max: 63.0 }
                angular_damping: 1.0
            }
            joints {
                name: "left_knee"
                parent: "left_thigh"
                child: "left_leg"
                parent_offset { z: -%s }
                child_offset { z: %s }
                rotation { z: -90.0 y: 0.0 }
                angle_limit { min: -91.0 max: -5.0 }
                angular_damping: 1.0
            }
            joints {
                name: "right_knee"
                parent: "right_thigh"
                child: "right_leg"
                parent_offset { z: -%s }
                child_offset { z: %s }
                rotation { z: -90.0 y: 0.0 }
                angle_limit { min: -91.0 max: -5.0 }
                angular_damping: 1.0
            }
            actuators {
                name: "left_hip"
                joint: "left_hip"
                strength: %s
                torque {}
            }
            actuators {
                name: "left_knee"
                joint: "left_knee"
                strength: %s
                torque {}
            }
            actuators {
                name: "right_hip"
                joint: "right_hip"
                strength: %s
                torque {}
            }
            actuators {
                name: "right_knee"
                joint: "right_knee"
                strength: %s
                torque {}
            }
            frozen {
                position { y: 1.0 }
                rotation { x: 1.0 z: 1.0 }
            }
            defaults {
                qps { name: "ground" pos { x: -%s y: %s z: 0 }}
                angles { name: "left_hip" angle { x: -20.0 } }
                angles { name: "right_hip" angle { x: 25.0 } }
                angles { name: "left_knee" angle { x: -40.0 } }
                angles { name: "right_knee" angle { x: -65.0 } }
            }
        """ % (
            TERRAIN_SIZE,
            terrain_map,
            HULL_X,
            HULL_Z,
            HULL_MASS,
            THIGH_X,
            THIGH_Z,
            LIMB_MASS,
            THIGH_X,
            THIGH_Z,
            LIMB_MASS,
            LEG_X,
            LEG_Z,
            LIMB_MASS,
            LEG_X,
            LEG_Z,
            LIMB_MASS,
            HIP_OFFSET_X,
            HULL_Z,
            -HIP_OFFSET_X,
            THIGH_Z,
            HIP_OFFSET_X,
            HULL_Z,
            -HIP_OFFSET_X,
            THIGH_Z,
            THIGH_Z,
            LEG_Z,
            THIGH_Z,
            LEG_Z,
            STRENGTH_HIP,
            STRENGTH_KNEE,
            STRENGTH_HIP,
            STRENGTH_KNEE,
            INITIAL_X,
            TERRAIN_SIZE / 2,
        )

        self.sys = brax.System(text_format.Parse(_SYSTEM_CONFIG, brax.Config()))

        self.body_sizes = [
            self._get_2d_box_size(body) for body in self.sys.config.bodies
        ]

        self.brax_step = jax.jit(self.sys.step)
        self.qp = self.sys.default_qp()

        self.qp, info = self.brax_step(self.qp, jp.array([0, 0, 0, 0]))
        self.state = self._get_state(info)
        self.renderer.render_step()
        if not return_info:
            return self.state
        else:
            return self.state, {}

    def step(self, action):
        action = jp.clip(action, -1, +1).astype(jp.float32)
        self.qp, info = self.brax_step(
            self.qp,
            jp.array(
                [
                    action[LEFT_HIP],
                    action[LEFT_KNEE],
                    action[RIGHT_HIP],
                    action[RIGHT_KNEE],
                ]
            ),
        )
        self.state = self._get_state(info)
        self.renderer.render_step()

        shaping = (
            3.5 * self.qp.pos[HULL][0]
        )  # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0 * abs(
            self.state[0]
        )  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        terminated = False
        if self.qp.pos[HULL][0] < -INITIAL_X / 2 or info.contact.vel[HULL, 2] != 0:
            reward = -100
            terminated = True
        if self.qp.pos[HULL][0] > TERRAIN_SIZE - INITIAL_X:
            terminated = True

        for a in action:
            reward -= 0.028 * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        return self.state, reward, terminated, False, {}

    def _get_vertices(self, x, y, w, h, rot):
        transformed_w, transformed_h = w * SCALE, h * SCALE
        angle = self._get_angle(rot)
        R = jp.array([[jp.cos(angle), jp.sin(angle)], [-jp.sin(angle), jp.cos(angle)]])
        centered_verts = jp.array(
            [
                transformed_w / 2 * jp.array([-1, 1, 1, -1]),
                transformed_h / 2 * jp.array([-1, -1, 1, 1]),
            ]
        )
        rot_centered_verts = (R @ centered_verts).T
        verts = rot_centered_verts + jp.array([x, y]) * SCALE
        return verts

    def render(self, mode="human"):
        if self.render_mode is not None:
            return self.renderer.get_renders()
        elif self.render_mode == "rgb_array":
            return image.render_array(self.sys, self.qp, 256, 256)
        else:
            return self._render(mode)

    def _render(self, mode="human"):
        assert mode in self.metadata["render_modes"]
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[phys2d]`"
            )

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            else:
                self.screen = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        scroll = int((self.qp.pos[HULL][0] + INITIAL_X) * SCALE + VIEWPORT_W / 4)
        render_offset_x = INITIAL_X * SCALE
        render_offset_y = VIEWPORT_H / 4

        self.surf = pygame.Surface((VIEWPORT_W + scroll - render_offset_x, VIEWPORT_H))
        self.surf.fill((215, 215, 255), self.surf.get_rect())

        def offset_coords(x, y):
            return jp.array((x + render_offset_x, y + render_offset_y))

        # lidar
        hull_pos = offset_coords(*self._to_2d(self.qp.pos[HULL] * SCALE))
        pygame.draw.polygon(
            self.surf,
            (255, 128, 128),
            [
                hull_pos,
                hull_pos - jp.array([0, VIEWPORT_H]),
                hull_pos
                + VIEWPORT_H * jp.array([jp.sin(9 * 1.5 / 10), -jp.cos(9 * 1.5 / 10)]),
            ],
        )

        # ground
        render_terrain_x, render_terrain_y = offset_coords(
            self.terrain_x * SCALE, self.terrain_y * SCALE
        )
        pygame.draw.polygon(
            self.surf,
            (102, 153, 76),
            jp.array(
                (
                    jp.concatenate(((0,), render_terrain_x, (TERRAIN_SIZE * SCALE,))),
                    jp.concatenate(((0,), render_terrain_y, (0,))),
                )
            ).T,
        )

        # flag
        flag_x = render_offset_x - VIEWPORT_W / 10
        pygame.draw.line(
            self.surf,
            color=(0, 0, 0),
            start_pos=(flag_x, render_offset_y),
            end_pos=(flag_x, render_offset_y + 50),
            width=1,
        )
        pygame.draw.polygon(
            self.surf,
            color=(230, 51, 0),
            points=[
                (flag_x, render_offset_y + 50),
                (flag_x, render_offset_y + 40),
                (flag_x + 25, render_offset_y + 45),
            ],
        )

        # bodies
        colors = [None] * 6
        colors[HULL] = (127, 51, 229)
        colors[LEFT_THIGH] = (178, 101, 152)
        colors[LEFT_LEG] = (178, 101, 152)
        colors[RIGHT_THIGH] = (128, 51, 102)
        colors[RIGHT_LEG] = (128, 51, 102)
        for body in (LEFT_THIGH, LEFT_LEG, RIGHT_THIGH, RIGHT_LEG, HULL):
            pos = self.qp.pos[body]
            rot = self.qp.rot[body]
            size = self.body_sizes[body]
            vertices = self._get_vertices(*self._to_2d(pos), *size, rot)
            pygame.draw.polygon(self.surf, colors[body], offset_coords(*vertices.T).T)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (-scroll, 0))

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode in {"rgb_array", "single_rgb_array"}:
            return jp.array(pygame.surfarray.pixels3d(self.surf)).transpose(1, 0, 2)[
                :, -VIEWPORT_W:
            ]

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    # Heurisic: suboptimal, have no notion of balance.
    env = BipedalWalker()
    env.reset()
    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1, 2, 3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    while True:
        s, r, terminated, truncated, info = env.step(a)
        total_reward += r
        if steps % 20 == 0 or terminated or truncated:
            print("\naction " + str([f"{x:+0.2f}" for x in a]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
            print("hull " + str([f"{x:+0.2f}" for x in s[0:4]]))
            print("leg0 " + str([f"{x:+0.2f}" for x in s[4:9]]))
            print("leg1 " + str([f"{x:+0.2f}" for x in s[9:14]]))
        steps += 1

        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5 * moving_leg
        supporting_s_base = 4 + 5 * supporting_leg

        hip_targ = [None, None]  # -0.8 .. +1.1
        knee_targ = [None, None]  # -0.6 .. +0.9
        hip_todo = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if state == STAY_ON_ONE_LEG:
            hip_targ[moving_leg] = 1.1
            knee_targ[moving_leg] = -0.6
            supporting_knee_angle += 0.03
            if s[2] > SPEED:
                supporting_knee_angle += 0.03
            supporting_knee_angle = min(supporting_knee_angle, SUPPORT_KNEE_ANGLE)
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base + 0] < 0.10:  # supporting leg is behind
                state = PUT_OTHER_DOWN
        if state == PUT_OTHER_DOWN:
            hip_targ[moving_leg] = +0.1
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base + 4]:
                state = PUSH_OFF
                supporting_knee_angle = min(s[moving_s_base + 2], SUPPORT_KNEE_ANGLE)
        if state == PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = +1.0
            if s[supporting_s_base + 2] > 0.88 or s[2] > 1.2 * SPEED:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        if hip_targ[0]:
            hip_todo[0] = 0.9 * (hip_targ[0] - s[4]) - 0.25 * s[5]
        if hip_targ[1]:
            hip_todo[1] = 0.9 * (hip_targ[1] - s[9]) - 0.25 * s[10]
        if knee_targ[0]:
            knee_todo[0] = 4.0 * (knee_targ[0] - s[6]) - 0.25 * s[7]
        if knee_targ[1]:
            knee_todo[1] = 4.0 * (knee_targ[1] - s[11]) - 0.25 * s[12]

        hip_todo[0] -= 0.9 * (0 - s[0]) - 1.5 * s[1]  # PID to keep head strait
        hip_todo[1] -= 0.9 * (0 - s[0]) - 1.5 * s[1]
        knee_todo[0] -= 15.0 * s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0 * s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5 * a, -1.0, 1.0)

        if terminated or truncated:
            break
