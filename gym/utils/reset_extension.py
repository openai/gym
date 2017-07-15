import numpy as np

def init_length(model, geom_name, body_name, joint_names, L, l):
    # model is mjmodel (self.model). body_name is the name of body attached directly to geom you are concedering.
    # L is numpy array of original geom's vector. l is the numpy array of geom's vector which you want new geom to be.
    # L and l must be parallel.
    # joint_names is a joint list that is attached directly to body_name.
    def parallel_assertion(L, l):
        unit_L = L / np.linalg.norm(L)
        unit_l = l / np.linalg.norm(l)
        assert unit_L.all() == unit_l.all()

    def find_parent(parent_list, parent_id):
        def get_id(id, parent_id):
            if id[0] == 0:
                return False
            elif id[0] == parent_id:
                return True
            else:
                return False

        tf = []
        for id in parent_list:
            bool = get_id(id, parent_id)
            tf.append(bool)

        tf[parent_id] = True
        return tf

    parallel_assertion(L,l)
    
    geom_name = geom_name.encode("utf-8")
    geom_idx = model.geom_names.index(geom_name)
    geom_size = np.array(model.geom_size)
    geom_size[geom_idx][1] = np.linalg.norm(l) / 2

    body_name = body_name.encode("utf-8")
    body_idx = model.body_names.index(body_name)
    body_pos = np.array(model.body_pos)
    parent_bools = find_parent(model.body_parentid, body_idx)
    
    joint_pos = np.array(model.jnt_pos)
    for joint_name in joint_names:
        joint_name = joint_name.encode("utf-8")
        joint_idx = model.joint_names.index(joint_name)
        joint_pos[joint_idx] -= (l-L) / 2

    for n, b in enumerate(parent_bools):
        if b:
            body_pos[n] += (l-L) / 2

    model.geom_size = geom_size
    model.body_pos = body_pos
    model.jnt_pos = joint_pos
