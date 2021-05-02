"""
Common functionality between the PDF and PS backends.
"""

import functools

import matplotlib as mpl
from matplotlib import _api
from .. import font_manager, ft2font
from ..afm import AFM
from ..backend_bases import RendererBase


@functools.lru_cache(50)
def _cached_get_afm_from_fname(fname):
    with open(fname, "rb") as fh:
        return AFM(fh)


class CharacterTracker:
    """
    Helper for font subsetting by the pdf and ps backends.

    Maintains a mapping of font paths to the set of character codepoints that
    are being used from that font.
    """

    def __init__(self):
        self.used = {}

    @_api.deprecated("3.3")
    @property
    def used_characters(self):
        d = {}
        for fname, chars in self.used.items():
            realpath, stat_key = mpl.cbook.get_realpath_and_stat(fname)
            d[stat_key] = (realpath, chars)
        return d

    def track(self, font, s):
        """Record that string *s* is being typeset using font *font*."""
        if isinstance(font, str):
            # Unused, can be removed after removal of track_characters.
            fname = font
        else:
            fname = font.fname
        self.used.setdefault(fname, set()).update(map(ord, s))

    # Not public, can be removed when pdf/ps merge_used_characters is removed.
    def merge(self, other):
        """Update self with a font path to character codepoints."""
        for fname, charset in other.items():
            self.used.setdefault(fname, set()).update(charset)


class RendererPDFPSBase(RendererBase):
    # The following attributes must be defined by the subclasses:
    # - _afm_font_dir
    # - _use_afm_rc_name

    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height

    def flipy(self):
        # docstring inherited
        return False  # y increases from bottom to top.

    def option_scale_image(self):
        # docstring inherited
        return True  # PDF and PS support arbitrary image scaling.

    def option_image_nocomposite(self):
        # docstring inherited
        # Decide whether to composite image based on rcParam value.
        return not mpl.rcParams["image.composite_image"]

    def get_canvas_width_height(self):
        # docstring inherited
        return self.width * 72.0, self.height * 72.0

    def get_text_width_height_descent(self, s, prop, ismath):
        # docstring inherited
        if ismath == "TeX":
            texmanager = self.get_texmanager()
            fontsize = prop.get_size_in_points()
            w, h, d = texmanager.get_text_width_height_descent(
                s, fontsize, renderer=self)
            return w, h, d
        elif ismath:
            # Circular import.
            from matplotlib.backends.backend_ps import RendererPS
            parse = self._text2path.mathtext_parser.parse(
                s, 72, prop,
                _force_standard_ps_fonts=(isinstance(self, RendererPS)
                                          and mpl.rcParams["ps.useafm"]))
            return parse.width, parse.height, parse.depth
        elif mpl.rcParams[self._use_afm_rc_name]:
            font = self._get_font_afm(prop)
            l, b, w, h, d = font.get_str_bbox_and_descent(s)
            scale = prop.get_size_in_points() / 1000
            w *= scale
            h *= scale
            d *= scale
            return w, h, d
        else:
            font = self._get_font_ttf(prop)
            font.set_text(s, 0.0, flags=ft2font.LOAD_NO_HINTING)
            w, h = font.get_width_height()
            d = font.get_descent()
            scale = 1 / 64
            w *= scale
            h *= scale
            d *= scale
            return w, h, d

    def _get_font_afm(self, prop):
        fname = font_manager.findfont(
            prop, fontext="afm", directory=self._afm_font_dir)
        return _cached_get_afm_from_fname(fname)

    def _get_font_ttf(self, prop):
        fname = font_manager.findfont(prop)
        font = font_manager.get_font(fname)
        font.clear()
        font.set_size(prop.get_size_in_points(), 72)
        return font
