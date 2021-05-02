# ----------------------------------------------------------------------------
# pyglet
# Copyright (c) 2006-2008 Alex Holkner
# Copyright (c) 2008-2020 pyglet contributors
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
"""Loading of 3D models.

A :py:class:`~pyglet.model.Model` is an instance of a 3D object.

The following example loads a ``"teapot.obj"`` model::

    import pyglet

    window = pyglet.window.Window()

    teapot = pyglet.model.load('teapot.obj')

    @window.event
    def on_draw():
        teapot.draw()

    pyglet.app.run()


You can also load models with :py:meth:`~pyglet.resource.model`.
See :py:mod:`~pyglet.resource` for more information.


Efficient Drawing
=================

As with Sprites or Text, Models can be added to a
:py:class:`~pyglet.graphics.Batch` for efficient drawing. This is
preferred to calling their ``draw`` methods individually.  To do this,
simply pass in a reference to the :py:class:`~pyglet.graphics.Batch`
instance when loading the Model::


    import pyglet

    window = pyglet.window.Window()
    batch = pyglet.graphics.Batch()

    teapot = pyglet.model.load('teapot.obj', batch=batch)

    @window.event
    def on_draw():
        batch.draw()

    pyglet.app.run()


.. versionadded:: 1.4
"""

import io

from pyglet.gl import *
from pyglet import graphics

from .codecs import ModelDecodeException
from .codecs import add_encoders, add_decoders, add_default_model_codecs
from .codecs import get_encoders, get_decoders


# Default matrix for models
_default_identity = [1.0, 0.0, 0.0, 0.0,
                     0.0, 1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0,
                     0.0, 0.0, 0.0, 1.0]


def load(filename, file=None, decoder=None, batch=None):
    """Load a 3D model from a file.

    :Parameters:
        `filename` : str
            Used to guess the model format, and to load the file if `file` is
            unspecified.
        `file` : file-like object or None
            Source of model data in any supported format.        
        `decoder` : ModelDecoder or None
            If unspecified, all decoders that are registered for the filename
            extension are tried. An exception is raised if no codecs are
            registered for the file extension, or if decoding fails.
        `batch` : Batch or None
            An optional Batch instance to add this model to.

    :rtype: :py:mod:`~pyglet.model.Model`
    """

    if not file:
        file = open(filename, 'rb')

    if not hasattr(file, 'seek'):
        file = io.BytesIO(file.read())

    try:
        if decoder:
            return decoder.decode(file, filename, batch)
        else:
            first_exception = None
            for decoder in get_decoders(filename):
                try:
                    model = decoder.decode(file, filename, batch)
                    return model
                except ModelDecodeException as e:
                    if (not first_exception or
                            first_exception.exception_priority < e.exception_priority):
                        first_exception = e
                    file.seek(0)

            if not first_exception:
                raise ModelDecodeException('No decoders are available for this model format.')
            raise first_exception
    finally:
        file.close()


class Model:
    """Instance of a 3D object.

    See the module documentation for usage.
    """

    def __init__(self, vertex_lists, groups, batch):
        """Create a model.

        :Parameters:
            `vertex_lists` : list
                A list of `~pyglet.graphics.VertexList` or
                `~pyglet.graphics.IndexedVertexList`.
            `groups` : list
                A list of `~pyglet.model.TexturedMaterialGroup`, or
                 `~pyglet.model.MaterialGroup`. Each group corresponds to
                 a vertex list in `vertex_lists` of the same index.
            `batch` : `~pyglet.graphics.Batch`
                Optional batch to add the model to. If no batch is provided,
                the model will maintain it's own internal batch.
        """
        self.groups = groups
        self.vertex_lists = vertex_lists
        self._batch = batch
        self._matrix = _default_identity

    @property
    def batch(self):
        """The graphics Batch that the Model belongs to.

        The Model can be migrated from one batch to another, or removed from
        a batch (for individual drawing). If not part of any batch, the Model
        will keep it's own internal batch. Note that batch migration can be
        an expensive operation.

        :type: :py:class:`pyglet.graphics.Batch`
        """
        return self._batch

    @batch.setter
    def batch(self, batch):
        if self._batch == batch:
            return

        if batch is None:
            batch = pyglet.graphics.Batch()

        for group, vlist in zip(self.groups, self.vertex_lists):
            self._batch.migrate(vlist, GL_TRIANGLES, group, batch)

        self._batch = batch

    @property
    def matrix(self):
        """Transformation matrix.

        A 4x4 matrix containing the desired transformation to
        apply. The data should be provided as a flat list or tuple.

        :type: list or tuple
        """
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        for group in self.groups:
            group.matrix[:] = matrix
        self._matrix = matrix

    def draw(self):
        """Draw the model.

        This is not recommended. See the module documentation
        for information on efficient drawing of multiple models.
        """
        self._batch.draw_subset(self.vertex_lists)


class Material:
    __slots__ = ("name", "diffuse", "ambient", "specular", "emission", "shininess", "texture_name")

    def __init__(self, name, diffuse, ambient, specular, emission, shininess, texture_name=None):
        self.name = name
        self.diffuse = (GLfloat * 4)(*diffuse)
        self.ambient = (GLfloat * 4)(*ambient)
        self.specular = (GLfloat * 4)(*specular)
        self.emission = (GLfloat * 4)(*emission)
        self.shininess = shininess
        self.texture_name = texture_name


class TexturedMaterialGroup(graphics.Group):

    def __init__(self, material, texture, matrix=None):
        super(TexturedMaterialGroup, self).__init__()
        self.material = material
        self.texture = texture
        self.matrix = (GLfloat * 16)(*matrix or _default_identity)

    def set_state(self, face=GL_FRONT_AND_BACK):
        glEnable(self.texture.target)
        glBindTexture(self.texture.target, self.texture.id)
        material = self.material
        glMaterialfv(face, GL_DIFFUSE, material.diffuse)
        glMaterialfv(face, GL_AMBIENT, material.ambient)
        glMaterialfv(face, GL_SPECULAR, material.specular)
        glMaterialfv(face, GL_EMISSION, material.emission)
        glMaterialf(face, GL_SHININESS, material.shininess)

        glPushMatrix()
        glMultMatrixf(self.matrix)

    def unset_state(self):
        glPopMatrix()
        glDisable(self.texture.target)
        glDisable(GL_COLOR_MATERIAL)

    def __eq__(self, other):
        # Do not consolidate Groups when adding to a Batch.
        # Matrix multiplications requires isolation.
        return False

    def __hash__(self):
        return hash((self.texture.id, self.texture.target))


class MaterialGroup(graphics.Group):

    def __init__(self, material, matrix=None):
        super(MaterialGroup, self).__init__()
        self.material = material
        self.matrix = (GLfloat * 16)(*matrix or _default_identity)

    def set_state(self, face=GL_FRONT_AND_BACK):
        glDisable(GL_TEXTURE_2D)
        material = self.material
        glMaterialfv(face, GL_DIFFUSE, material.diffuse)
        glMaterialfv(face, GL_AMBIENT, material.ambient)
        glMaterialfv(face, GL_SPECULAR, material.specular)
        glMaterialfv(face, GL_EMISSION, material.emission)
        glMaterialf(face, GL_SHININESS, material.shininess)

        glPushMatrix()
        glMultMatrixf(self.matrix)

    def unset_state(self):
        glPopMatrix()
        glDisable(GL_COLOR_MATERIAL)

    def __eq__(self, other):
        # Do not consolidate Groups when adding to a Batch.
        # Matrix multiplications requires isolation.
        return False

    def __hash__(self):
        material = self.material
        return hash((tuple(material.diffuse) + tuple(material.ambient) +
                     tuple(material.specular) + tuple(material.emission), material.shininess))


add_default_model_codecs()
