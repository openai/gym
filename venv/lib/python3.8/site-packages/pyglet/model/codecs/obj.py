# ----------------------------------------------------------------------------
# pyglet
# Copyright (c) 2006-2008 Alex Holkner
# Copyright (c) 2008-2021 pyglet contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------
import os

import pyglet

from pyglet.gl import GL_TRIANGLES

from .. import Model, Material, MaterialGroup, TexturedMaterialGroup
from . import ModelDecodeException, ModelDecoder


class Mesh:
    def __init__(self, name):
        self.name = name
        self.material = None

        self.indices = []
        self.vertices = []
        self.normals = []
        self.tex_coords = []
        self.colors = []


def load_material_library(filename):

    file = open(filename, 'r')

    name = None
    diffuse = [1.0, 1.0, 1.0]
    ambient = [1.0, 1.0, 1.0]
    specular = [1.0, 1.0, 1.0]
    emission = [0.0, 0.0, 0.0]
    shininess = 100.0
    opacity = 1.0
    texture_name = None

    matlib = {}

    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'newmtl':
            if name is not None:
                # save previous material
                for item in (diffuse, ambient, specular, emission):
                    item.append(opacity)
                matlib[name] = Material(name, diffuse, ambient, specular, emission, shininess, texture_name)
            name = values[1]

        elif name is None:
            raise ModelDecodeException('Expected "newmtl" in '.format(filename))

        try:
            if values[0] == 'Kd':
                diffuse = list(map(float, values[1:]))
            elif values[0] == 'Ka':
                ambient = list(map(float, values[1:]))
            elif values[0] == 'Ks':
                specular = list(map(float, values[1:]))
            elif values[0] == 'Ke':
                emission = list(map(float, values[1:]))
            elif values[0] == 'Ns':
                shininess = float(values[1])
            elif values[0] == 'd':
                opacity = float(values[1])
            elif values[0] == 'map_Kd':
                texture_name = values[1]

        except BaseException as ex:
            raise ModelDecodeException('Parsing error in {0}.'.format((filename, ex)))

    file.close()

    for item in (diffuse, ambient, specular, emission):
        item.append(opacity)

    matlib[name] = Material(name, diffuse, ambient, specular, emission, shininess, texture_name)

    return matlib


def parse_obj_file(filename, file=None):
    materials = {}
    mesh_list = []

    location = os.path.dirname(filename)

    if file is None:
        with open(filename, 'r') as f:
            file_contents = f.read()
    else:
        file_contents = file.read()

    if hasattr(file_contents, 'decode'):
        try:
            file_contents = file_contents.decode()
        except UnicodeDecodeError as e:
            raise ModelDecodeException("Unable to decode obj: {}".format(e))

    material = None
    mesh = None

    vertices = [[0., 0., 0.]]
    normals = [[0., 0., 0.]]
    tex_coords = [[0., 0.]]

    diffuse = [1.0, 1.0, 1.0, 1.0]
    ambient = [1.0, 1.0, 1.0, 1.0]
    specular = [1.0, 1.0, 1.0, 1.0]
    emission = [0.0, 0.0, 0.0, 1.0]
    shininess = 100.0

    default_material = Material("Default", diffuse, ambient, specular, emission, shininess)

    for line in file_contents.splitlines():

        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            vertices.append(list(map(float, values[1:4])))
        elif values[0] == 'vn':
            normals.append(list(map(float, values[1:4])))
        elif values[0] == 'vt':
            tex_coords.append(list(map(float, values[1:3])))

        elif values[0] == 'mtllib':
            material_abspath = os.path.join(location, values[1])
            materials = load_material_library(filename=material_abspath)            

        elif values[0] in ('usemtl', 'usemat'):
            material = materials.get(values[1])
            if mesh is not None:
                mesh.material = material

        elif values[0] == 'o':
            mesh = Mesh(name=values[1])
            mesh_list.append(mesh)

        elif values[0] == 'f':
            if mesh is None:
                mesh = Mesh(name='')
                mesh_list.append(mesh)
            if material is None:
                material = default_material
            if mesh.material is None:
                mesh.material = material

            # For fan triangulation, remember first and latest vertices
            n1 = None
            nlast = None
            t1 = None
            tlast = None
            v1 = None
            vlast = None

            for i, v in enumerate(values[1:]):
                v_i, t_i, n_i = (list(map(int, [j or 0 for j in v.split('/')])) + [0, 0])[:3]
                if v_i < 0:
                    v_i += len(vertices) - 1
                if t_i < 0:
                    t_i += len(tex_coords) - 1
                if n_i < 0:
                    n_i += len(normals) - 1

                mesh.normals += normals[n_i]
                mesh.tex_coords += tex_coords[t_i]
                mesh.vertices += vertices[v_i]

                if i >= 3:
                    # Triangulate
                    mesh.normals += n1 + nlast
                    mesh.tex_coords += t1 + tlast
                    mesh.vertices += v1 + vlast

                if i == 0:
                    n1 = normals[n_i]
                    t1 = tex_coords[t_i]
                    v1 = vertices[v_i]
                nlast = normals[n_i]
                tlast = tex_coords[t_i]
                vlast = vertices[v_i]

    return mesh_list


###################################################
#   Decoder definitions start here:
###################################################

class OBJModelDecoder(ModelDecoder):
    def get_file_extensions(self):
        return ['.obj']

    def decode(self, file, filename, batch):

        if not batch:
            batch = pyglet.graphics.Batch()

        mesh_list = parse_obj_file(filename=filename, file=file)

        vertex_lists = []
        groups = []

        for mesh in mesh_list:
            material = mesh.material
            if material.texture_name:
                texture = pyglet.resource.texture(material.texture_name)
                group = TexturedMaterialGroup(material, texture)
            else:
                group = MaterialGroup(material)
            groups.append(group)
            vertex_lists.append(batch.add(len(mesh.vertices) // 3,
                                          GL_TRIANGLES,
                                          group,
                                          ('v3f/static', mesh.vertices),
                                          ('n3f/static', mesh.normals),
                                          ('t2f/static', mesh.tex_coords)))

        return Model(vertex_lists=vertex_lists, groups=groups, batch=batch)


def get_decoders():
    return [OBJModelDecoder()]


def get_encoders():
    return []
