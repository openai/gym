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
import json
import struct

import pyglet

from pyglet.gl import GL_BYTE, GL_UNSIGNED_BYTE, GL_SHORT, GL_UNSIGNED_SHORT, GL_FLOAT
from pyglet.gl import GL_UNSIGNED_INT, GL_ELEMENT_ARRAY_BUFFER, GL_ARRAY_BUFFER, GL_TRIANGLES

from .. import Model, Material, MaterialGroup, TexturedMaterialGroup
from . import ModelDecodeException, ModelDecoder


# pyglet.graphics types
_pyglet_types = {
    GL_BYTE: 'b',
    GL_UNSIGNED_BYTE: 'B',
    GL_SHORT: 's',
    GL_UNSIGNED_SHORT: 'S',
    GL_UNSIGNED_INT: 'I',
    GL_FLOAT: 'f',
}

# struct module types
_struct_types = {
    GL_BYTE: 'b',
    GL_UNSIGNED_BYTE: 'B',
    GL_SHORT: 'h',
    GL_UNSIGNED_SHORT: 'H',
    GL_UNSIGNED_INT: 'I',
    GL_FLOAT: 'f',
}

# OpenGL type sizes
_component_sizes = {
    GL_BYTE: 1,
    GL_UNSIGNED_BYTE: 1,
    GL_SHORT: 2,
    GL_UNSIGNED_SHORT: 2,
    GL_UNSIGNED_INT: 4,
    GL_FLOAT: 4
}

_accessor_type_sizes = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16
}

_targets = {
    GL_ELEMENT_ARRAY_BUFFER: "element_array",
    GL_ARRAY_BUFFER: "array",
}

# GLTF to pyglet shorthand types:
_attributes = {
    'POSITION': 'v',
    'NORMAL': 'n',
    'TANGENT': None,
    'TEXCOORD_0': '0t',
    'TEXCOORD_1': '1t',
    'COLOR_0': 'c',
    'JOINTS_0': None,
    'WEIGHTS_0': None
}


class Buffer:
    # TODO: support GLB format
    # TODO: support data uris
    def __init__(self, length, uri):
        self._length = length
        self._uri = uri

    def read(self, offset, length):
        file = pyglet.resource.file(self._uri, 'rb')
        file.seek(offset)
        data = file.read(length)
        file.close()
        return data


class BufferView:
    def __init__(self, buffer, offset, length, target, stride):
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.target = target
        self.stride = stride


class Accessor:
    # TODO: support sparse accessors
    def __init__(self, buffer_view, offset, comp_type, count,
                 maximum, minimum, accessor_type, sparse):
        self.buffer_view = buffer_view
        self.offset = offset
        self.component_type = comp_type
        self.count = count
        self.maximum = maximum
        self.minimum = minimum
        self.type = accessor_type
        self.sparse = sparse
        self.size = _component_sizes[comp_type] * _accessor_type_sizes[accessor_type]

    def read(self):
        offset = self.offset + self.buffer_view.offset
        length = self.size * self.count
        stride = self.buffer_view.stride or 1
        # TODO: handle stride
        data = self.buffer_view.buffer.read(offset, length)
        return data


def parse_gltf_file(file, filename, batch):

    if file is None:
        file = pyglet.resource.file(filename, 'r')
    elif file.mode != 'r':
        file.close()
        file = pyglet.resource.file(filename, 'r')

    try:
        data = json.load(file)
    except json.JSONDecodeError:
        raise ModelDecodeException('Json error. Does not appear to be a valid glTF file.')
    finally:
        file.close()

    if 'asset' not in data:
        raise ModelDecodeException('Not a valid glTF file. Asset property not found.')
    else:
        if float(data['asset']['version']) < 2.0:
            raise ModelDecodeException('Only glTF 2.0+ models are supported')

    buffers = dict()
    buffer_views = dict()
    accessors = dict()
    materials = dict()

    for i, item in enumerate(data.get('buffers', [])):
        buffers[i] = Buffer(item['byteLength'], item['uri'])

    for i, item in enumerate(data.get('bufferViews', [])):
        buffer_index = item['buffer']
        buffer = buffers[buffer_index]
        offset = item.get('byteOffset', 0)
        length = item.get('byteLength')
        target = item.get('target')
        stride = item.get('byteStride', 1)
        buffer_views[i] = BufferView(buffer, offset, length, target, stride)

    for i, item in enumerate(data.get('accessors', [])):
        buf_view_index = item.get('bufferView')
        buf_view = buffer_views[buf_view_index]
        offset = item.get('byteOffset', 0)
        comp_type = item.get('componentType')
        count = item.get('count')
        maxi = item.get('max')
        mini = item.get('min')
        acc_type = item.get('type')
        sparse = item.get('sparse', None)
        accessors[i] = Accessor(buf_view, offset, comp_type, count, maxi, mini, acc_type, sparse)

    vertex_lists = []

    for mesh_data in data.get('meshes'):

        for primitive in mesh_data.get('primitives', []):
            indices = None
            attribute_list = []
            count = 0

            for attribute_type, i in primitive['attributes'].items():
                accessor = accessors[i]
                attrib = _attributes[attribute_type]
                if not attrib:
                    # TODO: Add support for these attribute types to pyglet
                    continue
                attrib_size = _accessor_type_sizes[accessor.type]
                pyglet_type = _pyglet_types[accessor.component_type]
                pyglet_fmt = "{0}{1}{2}".format(attrib, attrib_size, pyglet_type)

                count = accessor.count
                struct_fmt = str(count * attrib_size) + _struct_types[accessor.component_type]
                array = struct.unpack('<' + struct_fmt, accessor.read())

                attribute_list.append((pyglet_fmt, array))

            if 'indices' in primitive:
                indices_index = primitive.get('indices')
                accessor = accessors[indices_index]
                attrib_size = _accessor_type_sizes[accessor.type]
                fmt = str(accessor.count * attrib_size) + _struct_types[accessor.component_type]
                indices = struct.unpack('<' + fmt, accessor.read())

            # if 'material' in primitive:
            #     material_index = primitive.get('material')
            #     color = materials[material_index]
            #     attribute_list.append(('c4f', color * count))

            diffuse = [1.0, 1.0, 1.0]
            ambient = [1.0, 1.0, 1.0]
            specular = [1.0, 1.0, 1.0]
            emission = [0.0, 0.0, 0.0]
            shininess = 100.0
            opacity = 1.0
            material = Material("Default", diffuse, ambient, specular, emission, shininess, opacity)
            group = MaterialGroup(material=material)

            if indices:
                vlist = batch.add_indexed(count, GL_TRIANGLES, group, indices, *attribute_list)
            else:
                vlist = batch.add(count, GL_TRIANGLES, group, *attribute_list)

            vertex_lists.append(vlist)

    return vertex_lists


###################################################
#   Decoder definitions start here:
###################################################

class GLTFModelDecoder(ModelDecoder):
    def get_file_extensions(self):
        return ['.gltf']

    def decode(self, file, filename, batch):

        if not batch:
            batch = pyglet.graphics.Batch()

        vertex_lists = parse_gltf_file(file=file, filename=filename, batch=batch)
        textures = {}

        return Model(vertex_lists, textures, batch=batch)


def get_decoders():
    return [GLTFModelDecoder()]


def get_encoders():
    return []
