# """
# Implementation of a Jax-accelerated Acrobot environment.
# """
#
# from typing import Union
#
# import jax
# import jax.numpy as jnp
# import numpy as np
# from jax.random import PRNGKey
#
# from gym.functional import FuncEnv, StepReturn, StateType, ObsType
#
#
# class Acrobot(FuncEnv[jnp.ndarray, jnp.ndarray, int]):
#     dt = 0.2
#
#     LINK_LENGTH_1 = 1.0  # [m]
#     LINK_LENGTH_2 = 1.0  # [m]
#     LINK_MASS_1 = 1.0  #: [kg] mass of link 1
#     LINK_MASS_2 = 1.0  #: [kg] mass of link 2
#     LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
#     LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
#     LINK_MOI = 1.0  #: moments of inertia for both links
#
#     MAX_VEL_1 = 4 * jnp.pi
#     MAX_VEL_2 = 9 * jnp.pi
#
#     AVAIL_TORQUE = [-1.0, 0.0, +1]
#
#     torque_noise_max = 0.0
#
#     SCREEN_DIM = 500
#
#     #: use dynamics equations from the nips paper or the book
#     book_or_nips = "book"
#     action_arrow = None
#     domain_fig = None
#     actions_num = 3
#
#     init_low = -0.1
#     init_high = 0.1
#
#     def initial(self, rng: PRNGKey) -> jnp.ndarray:
#         return jax.random.uniform(rng, minval=self.init_low, maxval=self.init_high, shape=(4,))
#
#     def observation(self, state: jnp.ndarray) -> jnp.ndarray:
#         return jnp.array(
#             [jnp.cos(state[0]), jnp.sin(state[0]), jnp.cos(state[1]), jnp.sin(state[1]), state[2], state[3]]
#         )
#
#     def step(self, state: StateType, action: Union[int, jnp.ndarray], rng: PRNGKey) -> StepReturn:
#         s = state
#         a = action
#
#         torque = self.AVAIL_TORQUE[a]
#
#         # Removed condition because jax doesn't like them
#         torque += jax.random.uniform(rng, minval=-self.torque_noise_max, maxval=self.torque_noise_max)
#
#         # Now, augment the state with our force action so it can be passed to
#         # _dsdt
#         s_augmented = jnp.append(s, torque)
#
#         ns = rk4(self._dsdt, s_augmented, [0, self.dt])
#
#         ns[0] = wrap(ns[0], -pi, pi)
#         ns[1] = wrap(ns[1], -pi, pi)
#         ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
#         ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
#         self.state = ns
#         terminal = self._terminal()
#         reward = -1.0 if not terminal else 0.0
#
#         self.renderer.render_step()
#         return self._get_ob(), reward, terminal, {}
