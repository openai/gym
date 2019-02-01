import os
import numpy as np
from gym import utils
from gym.envs.robotics.hand import manipulate

# Ensure we get the path separator correct on windows
MANIPULATE_BLOCK_XML = os.path.join('hand', 'manipulate_block_touch_sensors_85.xml')
MANIPULATE_EGG_XML = os.path.join('hand', 'manipulate_egg_touch_sensors_85.xml')
MANIPULATE_PEN_XML = os.path.join('hand', 'manipulate_pen_touch_sensors_85.xml')


class ManipulateTouchSensorsEnv(manipulate.ManipulateEnv, utils.EzPickle):
    def __init__(
        self, model_path, target_position, target_rotation,
        target_position_range, reward_type, initial_qpos={},
        randomize_initial_position=True, randomize_initial_rotation=True,
        distance_threshold=0.01, rotation_threshold=0.1, n_substeps=20, relative_control=False,
        ignore_z_target_rotation=False, touch_visualisation="on_touch", touch_get_obs="boolean",
    ):
        """Initializes a new Hand manipulation environment with touch sensors.

        Args:
            touch_visualisation (string): how touch sensor sites are visualised
                - "always": always shows touch sensor sites
                - "on_touch": shows touch sensor sites only when touch values > 0
                - "off" or else: does not show touch sensor sites
            touch_get_obs (string): touch sensor readings
                - "boolean": returns 1 if touch sensor reading != 0.0 else 0
                - "log": returns log(x+1) touch sensor readings from self.sim.data.sensordata[]
                - "off" or else: does not add touch sensor readings to the observation

        """
        self.touch_visualisation = touch_visualisation
        self.touch_get_obs = touch_get_obs
        self._tsensor_id2name = {}
        self._tsensor_id2siteid = {}
        self._site_id2intial_rgba = {}  # dict for initial rgba values for debugging
        self.touch_color = [1, 0, 0, 0.4]
        self.notouch_color = [0, 0.5, 0, 0.3]

        manipulate.ManipulateEnv.__init__(
            self, model_path, target_position, target_rotation,
            target_position_range, reward_type, initial_qpos=initial_qpos,
            randomize_initial_position=randomize_initial_position, randomize_initial_rotation=randomize_initial_rotation,
            distance_threshold=distance_threshold, rotation_threshold=rotation_threshold, n_substeps=n_substeps, relative_control=relative_control,
            ignore_z_target_rotation=ignore_z_target_rotation,
        )
        utils.EzPickle.__init__(self)

        for k, v in self.sim.model._sensor_id2name.items():  # get touch sensor ids and their site names
            if 'TS' in v:
                self._tsensor_id2name[k] = v
                self._tsensor_id2siteid[k] = self.sim.model._site_name2id[v.replace('TS', 'T')]
                self._site_id2intial_rgba[self._tsensor_id2siteid[k]] = self.sim.model.site_rgba[self._tsensor_id2siteid[k]].copy()  # get initial rgba values


    def _get_obs(self):
        robot_qpos, robot_qvel = manipulate.robot_get_obs(self.sim)
        object_qvel = self.sim.data.get_joint_qvel('object:joint')
        achieved_goal = self._get_achieved_goal().ravel()  # this contains the object position + rotation
        touch_values = []  # get touch sensor readings. if there is one, set value to 1
        if self.touch_get_obs == 'boolean':
            for k, v in self._tsensor_id2name.items():
                value = 1.0 if self.sim.data.sensordata[k] != 0.0 else 0.0
                touch_values.append(value)
        elif self.touch_get_obs == 'log':
            for k, v in self._tsensor_id2name.items():
                value = self.sim.data.sensordata[k]
                touch_values.append(value)
            if len(touch_values) > 0:
                touch_values = np.log(np.array(touch_values) + 1.0)
        observation = np.concatenate([robot_qpos, robot_qvel, object_qvel, touch_values, achieved_goal])

        # set rgba values
        if self.touch_visualisation == 'always':
            for k, v in self._tsensor_id2name.items():
                self.sim.model.site_rgba[self._tsensor_id2siteid[k]] = self._site_id2intial_rgba[self._tsensor_id2siteid[k]].copy()
        elif self.touch_visualisation == 'on_touch':
            for k, v in self._tsensor_id2name.items():
                if self.sim.data.sensordata[k] != 0.0:
                    # self.sim.model.site_rgba[self._tsensor_id2siteid[k]] = self._site_id2intial_rgba[self._tsensor_id2siteid[k]].copy()
                    self.sim.model.site_rgba[self._tsensor_id2siteid[k]] = self.touch_color
                else:
                    self.sim.model.site_rgba[self._tsensor_id2siteid[k]] = self.notouch_color
        else:
            for k, v in self._tsensor_id2name.items():
                self.sim.model.site_rgba[self._tsensor_id2siteid[k]] = [0, 0, 0, 0]

        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.ravel().copy(),
        }

    def set_object_size(self, factor):  # changes size of the manipulated object
        for name in ['object', 'object_hidden', 'target']:
            id = self.sim.model._geom_name2id[name]
        self.sim.model.geom_size[id] *= factor


class HandBlockTouchSensorsEnv(ManipulateTouchSensorsEnv):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse'):
        super(HandBlockTouchSensorsEnv, self).__init__(
            model_path=MANIPULATE_BLOCK_XML, target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type)


class HandEggTouchSensorsEnv(ManipulateTouchSensorsEnv):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse'):
        super(HandEggTouchSensorsEnv, self).__init__(
            model_path=MANIPULATE_EGG_XML, target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type)


class HandPenTouchSensorsEnv(ManipulateTouchSensorsEnv):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse'):
        super(HandPenTouchSensorsEnv, self).__init__(
            model_path=MANIPULATE_PEN_XML, target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            randomize_initial_rotation=False, reward_type=reward_type,
            ignore_z_target_rotation=True, distance_threshold=0.05)
