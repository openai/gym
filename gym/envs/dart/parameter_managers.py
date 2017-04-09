__author__ = 'yuwenhao'

import numpy as np
from gym.envs.dart import dart_env

##############################################################################################################
################################  Hopper #####################################################################
##############################################################################################################

class hopperContactManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.6, 1.0] # friction range
        self.param_dim = 1

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        return np.array([friction_param])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)

    def resample_parameters(self):
        x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        self.set_simulator_parameters(x)

class hopperContactMassManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.6, 1.0] # friction range
        self.restitution_range = [0.0, 0.1]
        self.torso_mass_range = [3.0, 6.0]
        self.foot_mass_range = [3.0, 7.0]
        self.param_dim = 4

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_rest = self.simulator.dart_world.skeletons[0].bodynodes[0].restitution_coeff()
        restitution_param = (cur_rest - self.restitution_range[0]) / (self.restitution_range[1] - self.restitution_range[0])

        cur_mass = self.simulator.robot_skeleton.bodynodes[2].m
        mass_param = (cur_mass - self.torso_mass_range[0]) / (self.torso_mass_range[1] - self.torso_mass_range[0])

        cur_ft_mass = self.simulator.robot_skeleton.bodynodes[-1].m
        ft_mass_param = (cur_ft_mass - self.foot_mass_range[0]) / (self.foot_mass_range[1] - self.foot_mass_range[0])

        return np.array([friction_param, restitution_param, mass_param, ft_mass_param])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)

        restitution = x[1] * (self.restitution_range[1] - self.restitution_range[0]) + self.restitution_range[0]
        self.simulator.dart_world.skeletons[0].bodynodes[0].set_restitution_coeff(restitution)
        self.simulator.dart_world.skeletons[1].bodynodes[-1].set_restitution_coeff(1.0)

        mass = x[2] * (self.torso_mass_range[1] - self.torso_mass_range[0]) + self.torso_mass_range[0]
        self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)

        ft_mass = x[3] * (self.foot_mass_range[1] - self.foot_mass_range[0]) + self.foot_mass_range[0]
        self.simulator.robot_skeleton.bodynodes[-1].set_mass(ft_mass)

    def resample_parameters(self):
        #x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        x = np.random.normal(0, 0.2, self.param_dim) % 1
        self.set_simulator_parameters(x)

class hopperContactMassRoughnessManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.torso_mass_range = [3.0, 6.0]
        self.roughness_range = [-0.05, -0.02] # height of the obstacles
        self.param_dim = 3


    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_mass = self.simulator.robot_skeleton.bodynodes[2].m
        mass_param = (cur_mass - self.torso_mass_range[0]) / (self.torso_mass_range[1] - self.torso_mass_range[0])

        cq = self.simulator.dart_world.skeletons[0].q
        cur_height = cq[10]
        roughness_param = (cur_height - self.roughness_range[0]) / (self.roughness_range[1] - self.roughness_range[0])

        return np.array([friction_param, mass_param, roughness_param])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        for i in range(len(self.simulator.dart_world.skeletons[0].bodynodes)):
            self.simulator.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(friction)

        mass = x[1] * (self.torso_mass_range[1] - self.torso_mass_range[0]) + self.torso_mass_range[0]
        self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)

        obs_height = x[2] * (self.roughness_range[1] - self.roughness_range[0]) + self.roughness_range[0]
        cq = self.simulator.dart_world.skeletons[0].q
        cq[10] = obs_height
        cq[16] = obs_height
        cq[22] = obs_height
        cq[28] = obs_height
        cq[34] = obs_height
        cq[40] = obs_height
        self.simulator.dart_world.skeletons[0].q = cq

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        #x = np.random.normal(0, 0.2, self.param_dim) % 1
        x[2] = np.max([x[2]*1.5,1.0])
        self.set_simulator_parameters(x)

        if len(self.simulator.dart_world.skeletons[0].bodynodes) >= 7:
            cq = self.simulator.dart_world.skeletons[0].q
            pos = []
            pos.append(np.random.random()-0.5)
            for i in range(5):
                pos.append((pos[i] + 1.0/6.0)%1)
            np.random.shuffle(pos)
            cq[9] = pos[0]
            cq[15] = pos[1]
            cq[21] = pos[2]
            cq[27] = pos[3]
            cq[33] = pos[4]
            cq[39] = pos[5]
            self.simulator.dart_world.skeletons[0].q = cq

class hopperContactMassRoughnessManager_2d:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.torso_mass_range = [3.0, 6.0]
        self.roughness_range = [-0.05, -0.02] # height of the obstacles
        self.param_dim = 2


    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_mass = self.simulator.robot_skeleton.bodynodes[2].m
        mass_param = (cur_mass - self.torso_mass_range[0]) / (self.torso_mass_range[1] - self.torso_mass_range[0])

        return np.array([friction_param, mass_param])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        for i in range(len(self.simulator.dart_world.skeletons[0].bodynodes)):
            self.simulator.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(friction)

        mass = x[1] * (self.torso_mass_range[1] - self.torso_mass_range[0]) + self.torso_mass_range[0]
        self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)

        obs_height = x[2] * (self.roughness_range[1] - self.roughness_range[0]) + self.roughness_range[0]
        cq = self.simulator.dart_world.skeletons[0].q
        cq[10] = obs_height
        cq[16] = obs_height
        cq[22] = obs_height
        cq[28] = obs_height
        cq[34] = obs_height
        cq[40] = obs_height
        self.simulator.dart_world.skeletons[0].q = cq

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters())+1)
        #x = np.random.normal(0, 0.2, self.param_dim) % 1
        x[2]=1.0
        self.set_simulator_parameters(x)

        if len(self.simulator.dart_world.skeletons[0].bodynodes) >= 7:
            cq = self.simulator.dart_world.skeletons[0].q
            pos = []
            pos.append(np.random.random()-0.5)
            for i in range(5):
                pos.append((pos[i] + 1.0/6.0)%1)
            np.random.shuffle(pos)
            cq[9] = pos[0]
            cq[15] = pos[1]
            cq[21] = pos[2]
            cq[27] = pos[3]
            cq[33] = pos[4]
            cq[39] = pos[5]
            self.simulator.dart_world.skeletons[0].q = cq

class hopperRoughnessManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.roughness_range = [-0.05, -0.02] # height of the obstacles
        self.param_dim = 1


    def get_simulator_parameters(self):
        cq = self.simulator.dart_world.skeletons[0].q
        cur_height = cq[10]
        roughness_param = (cur_height - self.roughness_range[0]) / (self.roughness_range[1] - self.roughness_range[0])

        return np.array([roughness_param])

    def set_simulator_parameters(self, x):
        obs_height = x[0] * (self.roughness_range[1] - self.roughness_range[0]) + self.roughness_range[0]
        cq = self.simulator.dart_world.skeletons[0].q
        cq[10] = obs_height
        cq[16] = obs_height
        cq[22] = obs_height
        cq[28] = obs_height
        cq[34] = obs_height
        cq[40] = obs_height
        self.simulator.dart_world.skeletons[0].q = cq

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        #x = np.random.normal(0, 0.2, self.param_dim) % 1
        self.set_simulator_parameters(x)

        if len(self.simulator.dart_world.skeletons[0].bodynodes) >= 7:
            cq = self.simulator.dart_world.skeletons[0].q
            pos = []
            pos.append(np.random.random()-0.5)
            for i in range(5):
                pos.append((pos[i] + 1.0/6.0)%1)
            np.random.shuffle(pos)
            cq[9] = pos[0]
            cq[15] = pos[1]
            cq[21] = pos[2]
            cq[27] = pos[3]
            cq[33] = pos[4]
            cq[39] = pos[5]
            self.simulator.dart_world.skeletons[0].q = cq


class hopperContactAllMassManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.mass_range = [-1.0, 1.0]
        self.param_dim = 2
        self.initial_mass = []
        for i in range(4):
            self.initial_mass.append(simulator.robot_skeleton.bodynodes[2+i].m)

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_mass1 = self.simulator.robot_skeleton.bodynodes[2].m - self.initial_mass[0]
        mass_param1 = (cur_mass1 - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0])
        cur_mass2 = self.simulator.robot_skeleton.bodynodes[3].m - self.initial_mass[1]
        mass_param2 = (cur_mass2 - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0])
        cur_mass3 = self.simulator.robot_skeleton.bodynodes[4].m - self.initial_mass[2]
        mass_param3 = (cur_mass3 - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0])
        cur_mass4 = self.simulator.robot_skeleton.bodynodes[5].m - self.initial_mass[3]
        mass_param4 = (cur_mass4 - self.mass_range[0]) / (self.mass_range[1] - self.mass_range[0])

        return np.array([friction_param, mass_param1])#, mass_param2, mass_param3, mass_param4])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)

        mass = x[1] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0] + self.initial_mass[0]
        self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)

        '''mass = x[2] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0] + self.initial_mass[1]
        self.simulator.robot_skeleton.bodynodes[3].set_mass(mass)

        mass = x[3] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0] + self.initial_mass[2]
        self.simulator.robot_skeleton.bodynodes[4].set_mass(mass)

        mass = x[4] * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0] + self.initial_mass[3]
        self.simulator.robot_skeleton.bodynodes[5].set_mass(mass)'''

    def resample_parameters(self):
        #x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        x = np.random.normal(0, 0.2, 2) % 1
        self.set_simulator_parameters(x)

class hopperContactMassFootUpperLimitManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.torso_mass_range = [3.0, 6.0]
        self.limit_range = [-0.2, 0.2]
        self.param_dim = 2+1
        self.initial_up_limit = self.simulator.robot_skeleton.joints[-3].position_upper_limit(0)

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_mass = self.simulator.robot_skeleton.bodynodes[2].m
        mass_param = (cur_mass - self.torso_mass_range[0]) / (self.torso_mass_range[1] - self.torso_mass_range[0])

        # use upper limit of
        limit_diff1 = self.simulator.robot_skeleton.joints[-3].position_upper_limit(0) - self.initial_up_limits
        limit_diff1 = (limit_diff1 - self.limit_range[0]) / (self.limit_range[1] - self.limit_range[0])

        return np.array([friction_param, mass_param, limit_diff1])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        self.simulator.dart_world.skeletons[0].bodynodes[0].set_friction_coeff(friction)

        mass = x[1] * (self.torso_mass_range[1] - self.torso_mass_range[0]) + self.torso_mass_range[0]
        self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)

        limit_diff1 = x[2] * (self.limit_range[1] - self.limit_range[0]) + self.limit_range[0] + self.initial_up_limits

        self.simulator.robot_skeleton.joints[-3].set_position_upper_limit(0, limit_diff1)

    def resample_parameters(self):
        #x = np.random.uniform(0, 1, len(self.get_simulator_parameters()))
        x = np.random.normal(0, 0.2, 2) % 1
        self.set_simulator_parameters(x)

class hopperContactMassAllLimitManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.range = [0.3, 1.0] # friction range
        self.torso_mass_range = [3.0, 6.0]
        self.limit_range = [-0.3, 0.3]
        self.param_dim = 2+4
        self.initial_up_limits = []
        self.initial_low_limits = []
        for i in range(3):
            self.initial_up_limits.append(simulator.robot_skeleton.joints[-3+i].position_upper_limit(0))
            self.initial_low_limits.append(simulator.robot_skeleton.joints[-3+i].position_lower_limit(0))

    def get_simulator_parameters(self):
        cur_friction = self.simulator.dart_world.skeletons[0].bodynodes[0].friction_coeff()
        friction_param = (cur_friction - self.range[0]) / (self.range[1] - self.range[0])

        cur_mass = self.simulator.robot_skeleton.bodynodes[2].m
        mass_param = (cur_mass - self.torso_mass_range[0]) / (self.torso_mass_range[1] - self.torso_mass_range[0])

        # use upper limit of
        limit_diff1 = self.simulator.robot_skeleton.joints[-3].position_upper_limit(0) - self.initial_up_limits[0]
        limit_diff2 = self.simulator.robot_skeleton.joints[-2].position_upper_limit(0) - self.initial_up_limits[1]
        limit_diff3 = self.simulator.robot_skeleton.joints[-1].position_upper_limit(0) - self.initial_up_limits[2]
        limit_diff4 = self.simulator.robot_skeleton.joints[-1].position_lower_limit(0) - self.initial_low_limits[2]
        limit_diff1 = (limit_diff1 - self.limit_range[0]) / (self.limit_range[1] - self.limit_range[0])
        limit_diff2 = (limit_diff2 - self.limit_range[0]) / (self.limit_range[1] - self.limit_range[0])
        limit_diff3 = (limit_diff3 - self.limit_range[0]) / (self.limit_range[1] - self.limit_range[0])
        limit_diff4 = (limit_diff4 - self.limit_range[0]) / (self.limit_range[1] - self.limit_range[0])

        return np.array([friction_param, mass_param, limit_diff1, limit_diff2, limit_diff3, limit_diff4])

    def set_simulator_parameters(self, x):
        friction = x[0] * (self.range[1] - self.range[0]) + self.range[0]
        for i in range(len(self.simulator.dart_world.skeletons[0].bodynodes)):
            self.simulator.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(friction)

        mass = x[1] * (self.torso_mass_range[1] - self.torso_mass_range[0]) + self.torso_mass_range[0]
        self.simulator.robot_skeleton.bodynodes[2].set_mass(mass)

        limit_diff1 = x[2] * (self.limit_range[1] - self.limit_range[0]) + self.limit_range[0] + self.initial_up_limits[0]
        limit_diff2 = x[3] * (self.limit_range[1] - self.limit_range[0]) + self.limit_range[0] + self.initial_up_limits[1]
        limit_diff3 = x[4] * (self.limit_range[1] - self.limit_range[0]) + self.limit_range[0] + self.initial_up_limits[2]
        limit_diff4 = x[5] * (self.limit_range[1] - self.limit_range[0]) + self.limit_range[0] + self.initial_low_limits[2]

        self.simulator.robot_skeleton.joints[-3].set_position_upper_limit(0, limit_diff1)
        self.simulator.robot_skeleton.joints[-2].set_position_upper_limit(0, limit_diff2)
        self.simulator.robot_skeleton.joints[-1].set_position_upper_limit(0, limit_diff3)
        self.simulator.robot_skeleton.joints[-1].set_position_lower_limit(0, limit_diff4)

    def resample_parameters(self):
        x = np.random.uniform(-0.05, 1.05, len(self.get_simulator_parameters()))
        #x = np.random.normal(0, 0.2, 2) % 1
        self.set_simulator_parameters(x)

##############################################################################################################
##############################################################################################################




##############################################################################################################
################################  Hopper #####################################################################
##############################################################################################################