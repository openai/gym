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

"""Group multiple small images into larger textures.

This module is used by :py:mod:`pyglet.resource` to efficiently pack small
images into larger textures.  :py:class:`~pyglet.image.atlas.TextureAtlas` maintains one texture;
:py:class:`TextureBin` manages a collection of atlases of a given size.

Example usage::

    # Load images from disk
    car_image = pyglet.image.load('car.png')
    boat_image = pyglet.image.load('boat.png')

    # Pack these images into one or more textures
    bin = TextureBin()
    car_texture = bin.add(car_image)
    boat_texture = bin.add(boat_image)

The result of :py:meth:`TextureBin.add` is a :py:class:`TextureRegion`
containing the image. Once added, an image cannot be removed from a bin (or an 
atlas); nor can a list of images be obtained from a given bin or atlas -- it is 
the application's responsibility to keep track of the regions returned by the
``add`` methods.

.. versionadded:: 1.1
"""

import pyglet

from pyglet.gl import GL_RGBA


class AllocatorException(Exception):
    """The allocator does not have sufficient free space for the requested
    image size."""
    pass


class _Strip:
    __slots__ = 'x', 'y', 'max_height', 'y2'

    def __init__(self, y, max_height):
        self.x = 0
        self.y = y
        self.max_height = max_height
        self.y2 = y

    def add(self, width, height):
        assert width > 0 and height > 0
        assert height <= self.max_height

        x, y = self.x, self.y
        self.x += width
        self.y2 = max(self.y + height, self.y2)
        return x, y

    def compact(self):
        self.max_height = self.y2 - self.y


class Allocator:
    """Rectangular area allocation algorithm.

    Initialise with a given ``width`` and ``height``, then repeatedly
    call `alloc` to retrieve free regions of the area and protect that
    area from future allocations.

    `Allocator` uses a fairly simple strips-based algorithm.  It performs best
    when rectangles are allocated in decreasing height order.
    """
    __slots__ = 'width', 'height', 'strips', 'used_area'

    def __init__(self, width, height):
        """Create an `Allocator` of the given size.

        :Parameters:
            `width` : int
                Width of the allocation region.
            `height` : int
                Height of the allocation region.

        """
        assert width > 0 and height > 0
        self.width = width
        self.height = height
        self.strips = [_Strip(0, height)]
        self.used_area = 0

    def alloc(self, width, height):
        """Get a free area in the allocator of the given size.

        After calling `alloc`, the requested area will no longer be used.
        If there is not enough room to fit the given area `AllocatorException`
        is raised.

        :Parameters:
            `width` : int
                Width of the area to allocate.
            `height` : int
                Height of the area to allocate.

        :rtype: int, int
        :return: The X and Y coordinates of the bottom-left corner of the
            allocated region.
        """
        for strip in self.strips:
            if self.width - strip.x >= width and strip.max_height >= height:
                self.used_area += width * height
                return strip.add(width, height)

        if self.width >= width and self.height - strip.y2 >= height:
            self.used_area += width * height
            strip.compact()
            newstrip = _Strip(strip.y2, self.height - strip.y2)
            self.strips.append(newstrip)
            return newstrip.add(width, height)

        raise AllocatorException('No more space in %r for box %dx%d' % (self, width, height))

    def get_usage(self):
        """Get the fraction of area already allocated.

        This method is useful for debugging and profiling only.

        :rtype: float
        """
        return self.used_area / float(self.width * self.height)

    def get_fragmentation(self):
        """Get the fraction of area that's unlikely to ever be used, based on
        current allocation behaviour.

        This method is useful for debugging and profiling only.

        :rtype: float
        """
        # The total unused area in each compacted strip is summed.
        if not self.strips:
            return 0.0
        possible_area = self.strips[-1].y2 * self.width
        return 1.0 - self.used_area / float(possible_area)


class TextureAtlas:
    """Collection of images within a texture."""

    def __init__(self, width=2048, height=2048, border=False):
        """Create a texture atlas of the given size.

        :Parameters:
            `width` : int
                Width of the underlying texture.
            `height` : int
                Height of the underlying texture.
            `border` : bool
                If True, one pixel of blank space is left
                around each image added to the Atlas.

        """
        max_texture_size = pyglet.image.get_max_texture_size()
        width = min(width, max_texture_size)
        height = min(height, max_texture_size)

        self.texture = pyglet.image.Texture.create(width, height, GL_RGBA, rectangle=True)
        self.allocator = Allocator(width, height)
        self._border = 1 if border else 0

    def add(self, img):
        """Add an image to the atlas.

        This method will fail if the given image cannot be transferred
        directly to a texture (for example, if it is another texture).
        :py:class:`~pyglet.image.ImageData` is the usual image type for this method.

        `AllocatorException` will be raised if there is no room in the atlas
        for the image.

        :Parameters:
            `img` : `~pyglet.image.AbstractImage`
                The image to add.

        :rtype: :py:class:`~pyglet.image.TextureRegion`
        :return: The region of the atlas containing the newly added image.
        """
        b = self._border
        x, y = self.allocator.alloc(img.width + b*2, img.height + b*2)
        self.texture.blit_into(img, x+b, y+b, 0)
        return self.texture.get_region(x+b, y+b, img.width, img.height)


class TextureBin:
    """Collection of texture atlases.

    :py:class:`~pyglet.image.atlas.TextureBin` maintains a collection of texture atlases, and creates new
    ones as necessary to accommodate images added to the bin.
    """

    def __init__(self, texture_width=2048, texture_height=2048, border=False):
        """Create a texture bin for holding atlases of the given size.

        :Parameters:
            `texture_width` : int
                Width of texture atlases to create.
            `texture_height` : int
                Height of texture atlases to create.
            `border` : bool
                If True, one pixel of blank space is left
                around each image added to the Atlases.

        """
        max_texture_size = pyglet.image.get_max_texture_size()
        self.texture_width = min(texture_width, max_texture_size)
        self.texture_height = min(texture_height, max_texture_size)
        self.atlases = []
        self._border = border

    def add(self, img):
        """Add an image into this texture bin.

        This method calls `TextureAtlas.add` for the first atlas that has room
        for the image.

        `AllocatorException` is raised if the image exceeds the dimensions of
        ``texture_width`` and ``texture_height``.

        :Parameters:
            `img` : `~pyglet.image.AbstractImage`
                The image to add.

        :rtype: :py:class:`~pyglet.image.TextureRegion`
        :return: The region of an atlas containing the newly added image.
        """
        for atlas in list(self.atlases):
            try:
                return atlas.add(img)
            except AllocatorException:
                # Remove atlases that are no longer useful (this is so their
                # textures can later be freed if the images inside them get
                # collected).
                if img.width < 64 and img.height < 64:
                    self.atlases.remove(atlas)

        atlas = TextureAtlas(self.texture_width, self.texture_height)
        self.atlases.append(atlas)
        return atlas.add(img)
