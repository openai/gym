#
# The Python Imaging Library.
# $Id$
#
# standard mode descriptors
#
# History:
# 2006-03-20 fl   Added
#
# Copyright (c) 2006 by Secret Labs AB.
# Copyright (c) 2006 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#

# mode descriptor cache
_modes = None


class ModeDescriptor:
    """Wrapper for mode strings."""

    def __init__(self, mode, bands, basemode, basetype):
        self.mode = mode
        self.bands = bands
        self.basemode = basemode
        self.basetype = basetype

    def __str__(self):
        return self.mode


def getmode(mode):
    """Gets a mode descriptor for the given mode."""
    global _modes
    if not _modes:
        # initialize mode cache

        from . import Image

        modes = {}
        # core modes
        for m, (basemode, basetype, bands) in Image._MODEINFO.items():
            modes[m] = ModeDescriptor(m, bands, basemode, basetype)
        # extra experimental modes
        modes["RGBa"] = ModeDescriptor("RGBa", ("R", "G", "B", "a"), "RGB", "L")
        modes["LA"] = ModeDescriptor("LA", ("L", "A"), "L", "L")
        modes["La"] = ModeDescriptor("La", ("L", "a"), "L", "L")
        modes["PA"] = ModeDescriptor("PA", ("P", "A"), "RGB", "L")
        # mapping modes
        for i16mode in (
            "I;16",
            "I;16S",
            "I;16L",
            "I;16LS",
            "I;16B",
            "I;16BS",
            "I;16N",
            "I;16NS",
        ):
            modes[i16mode] = ModeDescriptor(i16mode, ("I",), "L", "L")
        # set global mode cache atomically
        _modes = modes
    return _modes[mode]
