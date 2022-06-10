import warnings
from typing import Optional

import brax
import jax
import numpy as np
from brax import jumpy as jp
from google.protobuf import text_format

import gym
from gym import spaces
from gym.error import DependencyNotInstalled
from gym.utils import EzPickle, colorize
from gym.utils.renderer import Renderer

VIEWPORT_W = 600
VIEWPORT_H = 400

INITIAL_HEIGHT = 30
CHUNKS = 11

MAIN_ENGINE_POWER = 80.0
SIDE_ENGINE_POWER = 26.0


class LunarLander(gym.Env, EzPickle):
    """
    ### Description
    This environment is a classic rocket trajectory optimization problem.
    According to Pontryagin's maximum principle, it is optimal to fire the
    engine at full throttle or turn it off. This is the reason why this
    environment has discrete actions: engine on or off.

    There are two environment versions: discrete or continuous.
    The landing pad is always at coordinates (0,0). The coordinates are the
    first two numbers in the state vector.
    Landing outside of the landing pad is possible. Fuel is infinite, so an agent
    can learn to fly and then land on its first attempt.

    To see a heuristic landing, run:
    ```
    python gym/envs/box2d/lunar_lander.py
    ```

    ### Action Space
    There are four discrete actions available: do nothing, fire left
    orientation engine, fire main engine, fire right orientation engine.

    ### Observation Space
    There are 8 states: the coordinates of the lander in `x` & `y`, its linear
    velocities in `x` & `y`, its angle, its angular velocity, and two booleans
    that represent whether each leg is in contact with the ground or not.

    ### Rewards
    Reward for moving from the top of the screen to the landing pad and coming
    to rest is about 100-140 points.
    If the lander moves away from the landing pad, it loses reward.
    If the lander crashes, it receives an additional -100 points. If it comes
    to rest, it receives an additional +100 points. Each leg with ground
    contact is +10 points.
    Firing the main engine is -0.3 points each frame. Firing the side engine
    is -0.03 points each frame. Solved is 200 points.

    ### Starting State
    The lander starts at the top center of the viewport with a random initial
    force applied to its center of mass.

    ### Episode Termination
    The episode finishes if:
    1) the lander crashes (the lander body gets in contact with the moon);
    2) the lander gets outside of the viewport (`x` coordinate is greater than 1);
    3) the lander comes to a halt

    ### Arguments
    To use to the _continuous_ environment, you need to specify the
    `continuous=True` argument like below:
    ```python
    import gym
    env = gym.make(
        "LunarLander-v2",
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    )
    ```
    If `continuous=True` is passed, continuous actions (corresponding to the throttle of the engines) will be used and the
    action space will be `Box(-1, +1, (2,), dtype=np.float32)`.
    The first coordinate of an action determines the throttle of the main engine, while the second
    coordinate specifies the throttle of the lateral boosters.
    Given an action `np.array([main, lateral])`, the main engine will be turned off completely if
    `main < 0` and the throttle scales affinely from 50% to 100% for `0 <= main <= 1` (in particular, the
    main engine doesn't work  with less than 50% power).
    Similarly, if `-0.5 < lateral < 0.5`, the lateral boosters will not fire at all. If `lateral < -0.5`, the left
    booster will fire, and if `lateral > 0.5`, the right booster will fire. Again, the throttle scales affinely
    from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively).

    `gravity` dictates the gravitational constant, this is bounded to be within 0 and -12.

    If `enable_wind=True` is passed, there will be wind effects applied to the lander.
    The wind is generated using the function `tanh(sin(2 k (t+C)) + sin(pi k (t+C)))`.
    `k` is set to 0.01.
    `C` is sampled randomly between -9999 and 9999.

    `wind_power` dictates the maximum magnitude of linear wind applied to the craft. The recommended value for `wind_power` is between 0.0 and 20.0.
    `turbulence_power` dictates the maximum magnitude of rotational wind applied to the craft. The recommended value for `turbulence_power` is between 0.0 and 2.0.

    ### Version History
    - v3: Replaced box2d with brax
    - v2: Count energy spent and in v0.24, added turbulance with wind power and turbulence_power parameters
    - v1: Legs contact with ground added in state vector; contact with ground
        give +10 reward points, and -10 if then lose contact; reward
        renormalized to 200; harder initial random push.
    - v0: Initial version

    <!-- ### References -->

    ### Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 100,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
        initial_push_power: float = 800,
    ):
        EzPickle.__init__(self)

        assert (
            -12.0 < gravity and gravity < 0.0
        ), f"gravity (current value: {gravity}) must be between -12 and 0"
        self.gravity = gravity

        if not 0.0 <= wind_power <= 20.0:
            warnings.warn(
                colorize(
                    f"WARN: wind_power value is recommended to be between 0.0 and 20.0, (current value: {wind_power})",
                    "yellow",
                ),
            )
        self.wind_power = wind_power

        if not 0.0 <= turbulence_power <= 2.0:
            warnings.warn(
                colorize(
                    f"WARN: turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {turbulence_power})",
                    "yellow",
                ),
            )
        self.turbulence_power = turbulence_power

        if not 0.0 <= initial_push_power <= 1000.0:
            warnings.warn(
                colorize(
                    f"WARN: initial_push_power value is recommended to be between 0.0 and 1000.0, (current value: {initial_push_power})",
                    "yellow",
                ),
            )
        self.initial_push_power = initial_push_power

        self.enable_wind = enable_wind
        self.wind_idx = self.np_random.integers(-9999, 9999)
        self.torque_idx = self.np_random.integers(-9999, 9999)

        self.screen = None
        self.clock = None

        self.continuous = continuous

        low = jp.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -1.5,
                -1.5,
                # velocity bounds is 5x rated speed
                -5.0,
                -5.0,
                -jp.pi,
                -5.0,
                -0.0,
                -0.0,
            ]
        ).astype(jp.float32)
        high = jp.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                1.5,
                1.5,
                # velocity bounds is 5x rated speed
                5.0,
                5.0,
                jp.pi,
                5.0,
                1.0,
                1.0,
            ]
        ).astype(jp.float32)

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(low, high)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)

    def _to_2d(self, arr):
        return arr[jp.array([0, 2])]

    def _get_angle(self, rot):
        # returns angle around y axis from rotation quaternion
        return 2 * jp.arctan2(rot[2], rot[0])

    def _get_2d_box_size(self, body):
        body_halfsizes = body.colliders[0].box.halfsize
        return 2 * jp.array([body_halfsizes.x, body_halfsizes.z])

    def _get_state(self, info):
        left_leg_ground_contact, right_leg_ground_contact = (
            info.contact.vel[1:-1, 2] != 0
        )
        scale_vec = self.SCALE / jp.array([VIEWPORT_W / 2, VIEWPORT_H * (3 / 4)])
        return np.array(
            [
                *jp.multiply(scale_vec, self._to_2d(self.qp.pos[0])),
                *jp.multiply(scale_vec, self._to_2d(self.qp.vel[0])),
                self._get_angle(self.qp.rot[0]),
                self.qp.ang[0][1] / 200,
                int(left_leg_ground_contact),
                int(right_leg_ground_contact),
            ],
            dtype=np.float32,
        )

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
        self.m_power = 0
        self.s_power = 0

        # terrain
        self.helipad_y = VIEWPORT_H / 4
        self.SCALE = (VIEWPORT_H - self.helipad_y) / INITIAL_HEIGHT
        self.render_height = self.np_random.uniform(
            0, VIEWPORT_H / 2, size=(CHUNKS + 1,)
        )
        self.render_height[CHUNKS // 2 - 2 : CHUNKS // 2 + 3] = self.helipad_y
        self.render_height = jp.array(
            [
                0.33
                * (
                    self.render_height[i - 1]
                    + self.render_height[i + 0]
                    + self.render_height[i + 1]
                )
                for i in range(CHUNKS)
            ]
        )

        brax_height = (self.render_height - self.helipad_y) / self.SCALE
        brax_heightmap = jp.repeat(brax_height, CHUNKS).tolist()

        _SYSTEM_CONFIG = """
          dt: 0.01
          substeps: 16
          friction: 0.5
          dynamics_mode: "pbd"
          gravity { z: %s }
          bodies {
            name: "lander" mass: 100
            colliders { box { halfsize { x: 1.7 y: 0.5 z: 1.4 }}}
            inertia { x: 1.0 y: 1.0 z: 1.0 }
          }
          bodies {
            name: "left_leg"
            colliders {
              box {
                halfsize { x: 0.2 y: 0.2 z: 0.95 }
              }
            }
            inertia { x: 1.0 y: 1.0 z: 1.0 }
            mass: 10.0
          }
          bodies {
            name: "right_leg"
            colliders {
              box {
                halfsize { x: 0.2 y: 0.2 z: 0.95 }
              }
            }
            inertia { x: 1.0 y: 1.0 z: 1.0 }
            mass: 10.0
          }
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
          joints {
            name: "left_joint"
            parent: "lander"
            child: "left_leg"
            parent_offset { x: -2.5 z: -0.8 }
            rotation { z: 90.0 y: 0.0 }
            angle_limit { min: 46.0 max: 62.5 }
          }
          joints {
            name: "right_joint"
            parent: "lander"
            child: "right_leg"
            parent_offset { x: 2.5 z: -0.8 }
            rotation { z: 90.0 y: -0.0 }
            angle_limit { min: -62.5 max: -46.0 }
          }
          actuators{
            name: "left_hinge"
            joint: "left_joint"
            strength: 600.0
            torque{
            }
          }
          actuators{
            name: "right_hinge"
            joint: "right_joint"
            strength: 600.0
            torque{
            }
          }
          forces {
            name: "thruster"
            body: "lander"
            strength: 300.0
            thruster {}
          }
          forces {
            name: "twister"
            body: "lander"
            strength: 200.0
            twister {}
          }
          frozen {
            position { y: 1.0 }
            rotation { x: 1.0 z: 1.0 }
          }
          defaults {
            qps { name: "lander" pos { z: %s }}
            qps { name: "ground" pos { x: -%s y: %s z: 0 }}
          }
        """ % (
            self.gravity,
            VIEWPORT_W / self.SCALE,
            brax_heightmap,
            INITIAL_HEIGHT,
            VIEWPORT_W / self.SCALE / 2,
            VIEWPORT_W / self.SCALE / 2,
        )

        self.sys = brax.System(text_format.Parse(_SYSTEM_CONFIG, brax.Config()))
        self.brax_step = jax.jit(self.sys.step)
        self.qp = self.sys.default_qp()

        self.lander_size = self._get_2d_box_size(self.sys.config.bodies[0])
        self.leg_size = self._get_2d_box_size(self.sys.config.bodies[1])

        self.qp, info = self.brax_step(
            self.qp,
            jp.concatenate(
                [
                    jp.array([-1, 1]),  # lander feet actuators
                    jp.array(
                        [
                            self.np_random.uniform(
                                -self.initial_push_power, self.initial_push_power
                            ),
                            0,
                            self.np_random.uniform(-self.initial_push_power, 0),
                        ]
                    ),
                    jp.zeros(3),
                ]
            ),
        )
        self.state = self._get_state(info)
        self.renderer.render_step()
        if not return_info:
            return self.state
        else:
            return self.state, {}

    def _lander_asleep(self):
        # the specific threshold depends on brax config parameters
        return jp.abs(self.qp.vel).sum() + jp.abs(self.qp.ang).sum() / 50 < 0.4

    def step(self, action):
        if self.continuous:
            action = jp.clip(action, -1, +1).astype(jp.float32)
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "

        # Engines
        self.m_power = 0
        self.s_power = 0
        if self.continuous:
            if action[0] > 0:
                self.m_power = (jp.clip(action[0], 0.0, 1.0) + 1.0) * 0.5
                assert 0.5 <= self.m_power <= 1.0
            self.s_power = action[1]
        else:
            if action == 1:
                self.s_power = 1
            elif action == 3:
                self.s_power = -1
            elif action == 2:
                self.m_power = 1

        thruster_vec = jp.array(
            [SIDE_ENGINE_POWER * self.s_power, 0, MAIN_ENGINE_POWER * self.m_power]
        )
        dispersion_vec = jp.array(
            [
                SIDE_ENGINE_POWER * self.np_random.uniform(-0.01, 0.01),
                0,
                MAIN_ENGINE_POWER * self.np_random.uniform(-0.01, 0.01),
            ]
        ) * jp.norm(thruster_vec)
        thruster_vec += dispersion_vec

        twister_vec = jp.zeros(3)

        # Wind
        if self.enable_wind and not (any(self.state[6:])):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = (
                jp.tanh(
                    jp.sin(0.02 * self.wind_idx)
                    + (jp.sin(jp.pi * 0.01 * self.wind_idx))
                )
                * self.wind_power
            )
            thruster_vec += wind_mag * jp.array(
                [jp.cos(self.state[4]), 0, 1 - jp.cos(self.state[4])]
            )
            self.wind_idx += 1

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = jp.tanh(
                jp.sin(0.02 * self.torque_idx)
                + (jp.sin(jp.pi * 0.01 * self.torque_idx))
            ) * (self.turbulence_power)
            twister_vec[1] = torque_mag
            self.torque_idx += 1

        self.qp, info = self.brax_step(
            self.qp,
            jp.concatenate([jp.array([-1, 1]), thruster_vec, twister_vec]),
        )
        self.state = self._get_state(info)
        self.renderer.render_step()

        reward = 0
        shaping = (
            -100 * jp.norm(self.state[:2])
            - 100 * jp.norm(self.state[2:4])
            - 100 * jp.abs(self.state[4])
            + 10 * self.state[6]
            + 10 * self.state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= (
            self.m_power * 0.30
        )  # less fuel spent is better, about -30 for heuristic landing
        reward -= abs(self.s_power) * 0.03

        done = False
        if info.contact.vel[0, 2] != 0 or jp.abs(self.state[0]) >= 1.0:
            done = True
            reward = -100

        if self._lander_asleep():
            done = True
            reward = +100
        return self.state, reward, done, {}

    def _get_vertices(self, x, y, w, h, rot):
        transformed_w, transformed_h = w * self.SCALE, h * self.SCALE
        angle = self._get_angle(rot)
        R = jp.array([[jp.cos(angle), jp.sin(angle)], [-jp.sin(angle), jp.cos(angle)]])
        centered_verts = jp.array(
            [
                transformed_w / 2 * jp.array([-1, 1, 1, -1]),
                transformed_h / 2 * jp.array([-1, -1, 1, 1]),
            ]
        )
        rot_centered_verts = (R @ centered_verts).T
        transformed_x = (x * self.SCALE) + VIEWPORT_W / 2
        transformed_y = (y * self.SCALE) + self.helipad_y
        verts = rot_centered_verts + jp.array([transformed_x, transformed_y])
        return verts

    def render(self, mode="human"):
        if self.render_mode is not None:
            return self.renderer.get_renders()
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

        self.surf = pygame.Surface(self.screen.get_size())
        self.surf.fill((0, 0, 0), self.surf.get_rect())

        # ground
        chunk_x = [VIEWPORT_W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        pygame.draw.polygon(
            self.surf,
            (255, 255, 255),
            ((0, 0),) + tuple(zip(chunk_x, self.render_height)) + ((VIEWPORT_W, 0),),
        )

        helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        for x in (helipad_x1, helipad_x2):
            pygame.draw.line(
                self.surf,
                color=(255, 255, 255),
                start_pos=(x, self.helipad_y),
                end_pos=(x, self.helipad_y + 50),
                width=1,
            )
            pygame.draw.polygon(
                self.surf,
                color=(204, 204, 0),
                points=[
                    (x, self.helipad_y + 50),
                    (x, self.helipad_y + 40),
                    (x + 25, self.helipad_y + 45),
                ],
            )

        # thrusters
        lander_pos = self.qp.pos[0]
        lander_rot = self.qp.rot[0]
        lander_vertices = self._get_vertices(
            *self._to_2d(lander_pos), *self.lander_size, lander_rot
        )

        def draw_thruster(corner_a, corner_b, length):
            dx, dy = corner_a - corner_b
            thruster_trail = jp.array([dy, -dx])
            for proportion, length_factor in zip((0.4, 0.5, 0.6), (0.3, 0.6, 0.3)):
                thruster_base = corner_a * proportion + corner_b * (1 - proportion)
                pygame.draw.line(
                    self.surf,
                    color=(220, 80, 80),
                    start_pos=(thruster_base + 0.1 * thruster_trail),
                    end_pos=(thruster_base + length * length_factor * thruster_trail),
                    width=2,
                )

        if self.m_power > 0:
            draw_thruster(lander_vertices[1], lander_vertices[0], self.m_power)
        if self.s_power > 0:
            draw_thruster(lander_vertices[0], lander_vertices[3], self.s_power)
        elif self.s_power < 0:
            draw_thruster(lander_vertices[2], lander_vertices[1], -self.s_power)

        # legs & lander
        for pos, rot in zip(self.qp.pos[1:-1], self.qp.rot[1:-1]):
            vertices = self._get_vertices(*self._to_2d(pos), *self.leg_size, rot)
            pygame.draw.polygon(self.surf, (128, 102, 230), vertices)
        pygame.draw.polygon(self.surf, (128, 102, 230), lander_vertices)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode in {"rgb_array", "single_rgb_array"}:
            return jp.array(pygame.surfarray.pixels3d(self.surf)).transpose(1, 0, 2)

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
            s[0] is the horizontal coordinate
            s[1] is the vertical coordinate
            s[2] is the horizontal speed
            s[3] is the vertical speed
            s[4] is the angle
            s[5] is the angular speed
            s[6] 1 if first leg has contact, else 0
            s[7] 1 if second leg has contact, else 0

    Returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = -s[0] * 0.5 - s[2] * 1.0  # angle should point towards center
    angle_targ = jp.clip(
        angle_targ, -0.4, 0.4
    )  # more than 0.4 radians (22 degrees) is bad
    hover_targ = 0.55 * jp.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(s[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = jp.array([20 * hover_todo - 1, 20 * angle_todo])
        a = jp.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > jp.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


def demo_heuristic_lander(env, seed=None, render=False):
    total_reward = 0
    steps = 0
    s = env.reset(seed=seed)
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open is False:
                break

        if steps % 20 == 0 or done:
            print("observations:", " ".join([f"{x:+0.2f}" for x in s]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if done:
            break
    if render:
        env.close()
    return total_reward


if __name__ == "__main__":
    demo_heuristic_lander(LunarLander(), render=True)
