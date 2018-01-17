import numpy as np
from gym import utils
from gym.envs.fetch import fetch_env
from gym import spaces


class FetchKickEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self):
        fetch_env.FetchEnv.__init__(self, 'kick.xml')
        utils.EzPickle.__init__(self)

    def _step(self, action):
        self.simulate(action)
        return 0, 0, 0, {}

    def _reset(self):
        pass

    @property
    def observation_space(self):
        return spaces.Box(-np.ones(2), np.ones(2))

    @property
    def initial_qpos(self):
        init_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'geom0:slide0': 0.7,
            'geom0:slide1': 0.3,
            'geom0:slide2': 0.0,
            'geom1:slide0': 1.703020558521492,
            'geom1:slide1': 1.0816411287521643,
            'geom1:slide2': 0.4,
        }
        return init_qpos

    def initial_setup(self):
        fetch_env.FetchEnv.initial_setup(self)

    # def _step(self, action):
    #     xposbefore = self.sim.data.qpos[0]
    #     #self.do_simulation(action, self.frame_skip)
    #     xposafter = self.sim.data.qpos[0]
    #     ob = self._get_obs()
    #     reward_ctrl = - 0.1 * np.square(action).sum()
    #     reward_run = (xposafter - xposbefore)/self.dt
    #     reward = reward_ctrl + reward_run
    #     done = False
    #     return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    # def _get_obs(self):
    #     return np.concatenate([
    #         self.sim.data.qpos.flat[1:],
    #         self.sim.data.qvel.flat,
    #     ])

    # def reset_model(self):
    #     # Move into correct initial configuration.
    #     init_qpos = {
    #         'robot0:slide0': 0.05,
    #         'robot0:slide1': 0.48,
    #         'robot0:slide2': 0.0,
    #         'geom0:slide0': 0.7,
    #         'geom0:slide1': 0.3,
    #         'geom0:slide2': 0.0,
    #         'geom1:slide0': 1.703020558521492,
    #         'geom1:slide1': 1.0816411287521643,
    #         'geom1:slide2': 0.4
    #     }
    #     for name, value in init_qpos.items():
    #         self.sim.data.set_joint_qpos(name, value)

    #     # Places mocap where related bodies are.
    #     if self.sim.model.nmocap > 0 and self.sim.model.eq_data is not None:
    #         for i in range(self.sim.model.eq_data.shape[0]):
    #             if self.sim.model.eq_type[i] == const.EQ_WELD:
    #                 self.sim.model.eq_data[i, :] = np.array(
    #                     [0., 0., 0., 1., 0., 0., 0.])

    #     self.sim.forward()

    #     # Move gripper
    #     gripper_extra_height = 0.
    #     gripper_target = np.array([-0.498, 0.005, -0.431 + gripper_extra_height]) + self.sim.data.get_site_xpos(
    #         'robot0:grip')
    #     for _ in range(1000):
    #         action = np.zeros(4, np.float32)
    #         action[:3] = 0.01 * np.sign(gripper_target - self.sim.data.get_site_xpos('robot0:grip'))
    #         #action[3] = float(not self.block_gripper)
    #         # TODO: proper support for setting actions
    #         self._set_action(sim, action)
    #         sim.step()

    #     return self._get_obs()

    # def viewer_setup(self):
    #     self.viewer.cam.distance = self.model.stat.extent * 0.5