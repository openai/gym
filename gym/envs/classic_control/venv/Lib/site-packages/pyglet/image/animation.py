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

"""2D Animations

Animations can be used by the :py:class:`~pyglet.sprite.Sprite` class in place
of static images. They are essentially containers for individual image frames,
with a duration per frame. They can be infinitely looping, or stop at the last
frame. You can load Animations from disk, such as from GIF files::

    ani = pyglet.resource.animation('walking.gif')
    sprite = pyglet.sprite.Sprite(img=ani)

Alternatively, you can create your own Animations from a sequence of images
by using the :py:meth:`~Animation.from_image_sequence` method::

    images = [pyglet.resource.image('walk_a.png')
              pyglet.resource.image('walk_b.png')
              pyglet.resource.image('walk_c.png')]

    ani = pyglet.image.Animation.from_image_sequence(images, duration=0.1, loop=True)

You can also use an :py:class:`pyglet.image.ImageGrid`, which is iterable::

    sprite_sheet = pyglet.resource.image('my_sprite_sheet.png')
    image_grid = pyglet.image.ImageGrid(sprite_sheet, rows=1, columns=5)

    ani = pyglet.image.Animation.from_image_sequence(image_grid, duration=0.1)

In the above examples, all of the Animation Frames have the same duration.
If you wish to adjust this, you can manually create the Animation from a list of
:py:class:`~AnimationFrame`::

    image_a = pyglet.resource.image('walk_a.png')
    image_b = pyglet.resource.image('walk_b.png')
    image_c = pyglet.resource.image('walk_c.png')

    frame_a = pyglet.image.AnimationFrame(image_a, duration=0.1)
    frame_b = pyglet.image.AnimationFrame(image_b, duration=0.2)
    frame_c = pyglet.image.AnimationFrame(image_c, duration=0.1)

    ani = pyglet.image.Animation(frames=[frame_a, frame_b, frame_c])

"""


class Animation:
    """Sequence of images with timing information.

    If no frames of the animation have a duration of ``None``, the animation
    loops continuously; otherwise the animation stops at the first frame with
    duration of ``None``.

    :Ivariables:
        `frames` : list of `~pyglet.image.AnimationFrame`
            The frames that make up the animation.

    """

    def __init__(self, frames):
        """Create an animation directly from a list of frames.

        :Parameters:
            `frames` : list of `~pyglet.image.AnimationFrame`
                The frames that make up the animation.

        """
        assert len(frames)
        self.frames = frames

    def add_to_texture_bin(self, texture_bin):
        """Add the images of the animation to a :py:class:`~pyglet.image.atlas.TextureBin`.

        The animation frames are modified in-place to refer to the texture bin
        regions.

        :Parameters:
            `texture_bin` : `~pyglet.image.atlas.TextureBin`
                Texture bin to upload animation frames into.

        """
        for frame in self.frames:
            frame.image = texture_bin.add(frame.image)

    def get_transform(self, flip_x=False, flip_y=False, rotate=0):
        """Create a copy of this animation applying a simple transformation.

        The transformation is applied around the image's anchor point of
        each frame.  The texture data is shared between the original animation
        and the transformed animation.

        :Parameters:
            `flip_x` : bool
                If True, the returned animation will be flipped horizontally.
            `flip_y` : bool
                If True, the returned animation will be flipped vertically.
            `rotate` : int
                Degrees of clockwise rotation of the returned animation.  Only
                90-degree increments are supported.

        :rtype: :py:class:`~pyglet.image.Animation`
        """
        frames = [AnimationFrame(frame.image.get_texture().get_transform(flip_x, flip_y, rotate),
                                 frame.duration) for frame in self.frames]
        return Animation(frames)

    def get_duration(self):
        """Get the total duration of the animation in seconds.

        :rtype: float
        """
        return sum([frame.duration for frame in self.frames if frame.duration is not None])

    def get_max_width(self):
        """Get the maximum image frame width.

        This method is useful for determining texture space requirements: due
        to the use of ``anchor_x`` the actual required playback area may be
        larger.

        :rtype: int
        """
        return max([frame.image.width for frame in self.frames])

    def get_max_height(self):
        """Get the maximum image frame height.

        This method is useful for determining texture space requirements: due
        to the use of ``anchor_y`` the actual required playback area may be
        larger.

        :rtype: int
        """
        return max([frame.image.height for frame in self.frames])

    @classmethod
    def from_image_sequence(cls, sequence, period, loop=True):
        """Create an animation from a list of images and a constant framerate.

        :Parameters:
            `sequence` : list of `~pyglet.image.AbstractImage`
                Images that make up the animation, in sequence.
            `period` : float
                Number of seconds to display each image.
            `loop` : bool
                If True, the animation will loop continuously.

        :rtype: :py:class:`~pyglet.image.Animation`
        """
        frames = [AnimationFrame(image, period) for image in sequence]
        if not loop:
            frames[-1].duration = None
        return cls(frames)

    def __repr__(self):
        return "Animation(frames={0})".format(len(self.frames))


class AnimationFrame:
    """A single frame of an animation."""

    __slots__ = 'image', 'duration'

    def __init__(self, image, duration):
        """Create an animation frame from an image.

        :Parameters:
            `image` : `~pyglet.image.AbstractImage`
                The image of this frame.
            `duration` : float
                Number of seconds to display the frame, or ``None`` if it is
                the last frame in the animation.

        """
        self.image = image
        self.duration = duration

    def __repr__(self):
        return "AnimationFrame({0}, duration={1})".format(self.image, self.duration)
