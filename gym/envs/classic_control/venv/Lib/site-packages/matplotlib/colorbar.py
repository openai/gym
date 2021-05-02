"""
Colorbars are a visualization of the mapping from scalar values to colors.
In Matplotlib they are drawn into a dedicated `~.axes.Axes`.

.. note::
   Colorbars are typically created through `.Figure.colorbar` or its pyplot
   wrapper `.pyplot.colorbar`, which use `.make_axes` and `.Colorbar`
   internally.

   As an end-user, you most likely won't have to call the methods or
   instantiate the classes in this module explicitly.

:class:`ColorbarBase`
    The base class with full colorbar drawing functionality.
    It can be used as-is to make a colorbar for a given colormap;
    a mappable object (e.g., image) is not needed.

:class:`Colorbar`
    On top of `.ColorbarBase` this connects the colorbar with a
    `.ScalarMappable` such as an image or contour plot.

:func:`make_axes`
    Create an `~.axes.Axes` suitable for a colorbar. This functions can be
    used with figures containing a single axes or with freely placed axes.

:func:`make_axes_gridspec`
    Create a `~.SubplotBase` suitable for a colorbar. This function should
    be used for adding a colorbar to a `.GridSpec`.
"""

import copy
import logging
import textwrap

import numpy as np

import matplotlib as mpl
from matplotlib import _api, collections, cm, colors, contour, ticker
import matplotlib.artist as martist
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
from matplotlib import docstring

_log = logging.getLogger(__name__)

_make_axes_param_doc = """
location : None or {'left', 'right', 'top', 'bottom'}
    The location, relative to the parent axes, where the colorbar axes
    is created.  It also determines the *orientation* of the colorbar
    (colorbars on the left and right are vertical, colorbars at the top
    and bottom are horizontal).  If None, the location will come from the
    *orientation* if it is set (vertical colorbars on the right, horizontal
    ones at the bottom), or default to 'right' if *orientation* is unset.
orientation : None or {'vertical', 'horizontal'}
    The orientation of the colorbar.  It is preferable to set the *location*
    of the colorbar, as that also determines the *orientation*; passing
    incompatible values for *location* and *orientation* raises an exception.
fraction : float, default: 0.15
    Fraction of original axes to use for colorbar.
shrink : float, default: 1.0
    Fraction by which to multiply the size of the colorbar.
aspect : float, default: 20
    Ratio of long to short dimensions.
"""
_make_axes_other_param_doc = """
pad : float, default: 0.05 if vertical, 0.15 if horizontal
    Fraction of original axes between colorbar and new image axes.
anchor : (float, float), optional
    The anchor point of the colorbar axes.
    Defaults to (0.0, 0.5) if vertical; (0.5, 1.0) if horizontal.
panchor : (float, float), or *False*, optional
    The anchor point of the colorbar parent axes. If *False*, the parent
    axes' anchor will be unchanged.
    Defaults to (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal.
"""

_colormap_kw_doc = """

    ============  ====================================================
    Property      Description
    ============  ====================================================
    *extend*      {'neither', 'both', 'min', 'max'}
                  If not 'neither', make pointed end(s) for out-of-
                  range values.  These are set for a given colormap
                  using the colormap set_under and set_over methods.
    *extendfrac*  {*None*, 'auto', length, lengths}
                  If set to *None*, both the minimum and maximum
                  triangular colorbar extensions with have a length of
                  5% of the interior colorbar length (this is the
                  default setting). If set to 'auto', makes the
                  triangular colorbar extensions the same lengths as
                  the interior boxes (when *spacing* is set to
                  'uniform') or the same lengths as the respective
                  adjacent interior boxes (when *spacing* is set to
                  'proportional'). If a scalar, indicates the length
                  of both the minimum and maximum triangular colorbar
                  extensions as a fraction of the interior colorbar
                  length. A two-element sequence of fractions may also
                  be given, indicating the lengths of the minimum and
                  maximum colorbar extensions respectively as a
                  fraction of the interior colorbar length.
    *extendrect*  bool
                  If *False* the minimum and maximum colorbar extensions
                  will be triangular (the default). If *True* the
                  extensions will be rectangular.
    *spacing*     {'uniform', 'proportional'}
                  Uniform spacing gives each discrete color the same
                  space; proportional makes the space proportional to
                  the data interval.
    *ticks*       *None* or list of ticks or Locator
                  If None, ticks are determined automatically from the
                  input.
    *format*      None or str or Formatter
                  If None, `~.ticker.ScalarFormatter` is used.
                  If a format string is given, e.g., '%.3f', that is used.
                  An alternative `~.ticker.Formatter` may be given instead.
    *drawedges*   bool
                  Whether to draw lines at color boundaries.
    *label*       str
                  The label on the colorbar's long axis.
    ============  ====================================================

    The following will probably be useful only in the context of
    indexed colors (that is, when the mappable has norm=NoNorm()),
    or other unusual circumstances.

    ============   ===================================================
    Property       Description
    ============   ===================================================
    *boundaries*   None or a sequence
    *values*       None or a sequence which must be of length 1 less
                   than the sequence of *boundaries*. For each region
                   delimited by adjacent entries in *boundaries*, the
                   colormapped to the corresponding value in values
                   will be used.
    ============   ===================================================

"""

docstring.interpd.update(colorbar_doc="""
Add a colorbar to a plot.

Parameters
----------
mappable
    The `matplotlib.cm.ScalarMappable` (i.e., `~matplotlib.image.AxesImage`,
    `~matplotlib.contour.ContourSet`, etc.) described by this colorbar.
    This argument is mandatory for the `.Figure.colorbar` method but optional
    for the `.pyplot.colorbar` function, which sets the default to the current
    image.

    Note that one can create a `.ScalarMappable` "on-the-fly" to generate
    colorbars not attached to a previously drawn artist, e.g. ::

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

cax : `~matplotlib.axes.Axes`, optional
    Axes into which the colorbar will be drawn.

ax : `~matplotlib.axes.Axes`, list of Axes, optional
    One or more parent axes from which space for a new colorbar axes will be
    stolen, if *cax* is None.  This has no effect if *cax* is set.

use_gridspec : bool, optional
    If *cax* is ``None``, a new *cax* is created as an instance of Axes.  If
    *ax* is an instance of Subplot and *use_gridspec* is ``True``, *cax* is
    created as an instance of Subplot using the :mod:`~.gridspec` module.

Returns
-------
colorbar : `~matplotlib.colorbar.Colorbar`
    See also its base class, `~matplotlib.colorbar.ColorbarBase`.

Notes
-----
Additional keyword arguments are of two kinds:

  axes properties:
%s
%s
  colorbar properties:
%s

If *mappable* is a `~.contour.ContourSet`, its *extend* kwarg is included
automatically.

The *shrink* kwarg provides a simple way to scale the colorbar with respect
to the axes. Note that if *cax* is specified, it determines the size of the
colorbar and *shrink* and *aspect* kwargs are ignored.

For more precise control, you can manually specify the positions of
the axes objects in which the mappable and the colorbar are drawn.  In
this case, do not use any of the axes properties kwargs.

It is known that some vector graphics viewers (svg and pdf) renders white gaps
between segments of the colorbar.  This is due to bugs in the viewers, not
Matplotlib.  As a workaround, the colorbar can be rendered with overlapping
segments::

    cbar = colorbar()
    cbar.solids.set_edgecolor("face")
    draw()

However this has negative consequences in other circumstances, e.g. with
semi-transparent images (alpha < 1) and colorbar extensions; therefore, this
workaround is not used by default (see issue #1188).
""" % (textwrap.indent(_make_axes_param_doc, "    "),
       textwrap.indent(_make_axes_other_param_doc, "    "),
       _colormap_kw_doc))

# Deprecated since 3.4.
colorbar_doc = docstring.interpd.params["colorbar_doc"]
colormap_kw_doc = _colormap_kw_doc
make_axes_kw_doc = _make_axes_param_doc + _make_axes_other_param_doc


def _set_ticks_on_axis_warn(*args, **kw):
    # a top level function which gets put in at the axes'
    # set_xticks and set_yticks by ColorbarBase.__init__.
    _api.warn_external("Use the colorbar set_ticks() method instead.")


class _ColorbarAutoLocator(ticker.MaxNLocator):
    """
    AutoLocator for Colorbar

    This locator is just a `.MaxNLocator` except the min and max are
    clipped by the norm's min and max (i.e. vmin/vmax from the
    image/pcolor/contour object).  This is necessary so ticks don't
    extrude into the "extend regions".
    """

    def __init__(self, colorbar):
        """
        This ticker needs to know the *colorbar* so that it can access
        its *vmin* and *vmax*.  Otherwise it is the same as
        `~.ticker.AutoLocator`.
        """

        self._colorbar = colorbar
        nbins = 'auto'
        steps = [1, 2, 2.5, 5, 10]
        super().__init__(nbins=nbins, steps=steps)

    def tick_values(self, vmin, vmax):
        # flip if needed:
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        vmin = max(vmin, self._colorbar.norm.vmin)
        vmax = min(vmax, self._colorbar.norm.vmax)
        ticks = super().tick_values(vmin, vmax)
        rtol = (vmax - vmin) * 1e-10
        return ticks[(ticks >= vmin - rtol) & (ticks <= vmax + rtol)]


class _ColorbarAutoMinorLocator(ticker.AutoMinorLocator):
    """
    AutoMinorLocator for Colorbar

    This locator is just a `.AutoMinorLocator` except the min and max are
    clipped by the norm's min and max (i.e. vmin/vmax from the
    image/pcolor/contour object).  This is necessary so that the minorticks
    don't extrude into the "extend regions".
    """

    def __init__(self, colorbar, n=None):
        """
        This ticker needs to know the *colorbar* so that it can access
        its *vmin* and *vmax*.
        """
        self._colorbar = colorbar
        self.ndivs = n
        super().__init__(n=None)

    def __call__(self):
        vmin = self._colorbar.norm.vmin
        vmax = self._colorbar.norm.vmax
        ticks = super().__call__()
        rtol = (vmax - vmin) * 1e-10
        return ticks[(ticks >= vmin - rtol) & (ticks <= vmax + rtol)]


class _ColorbarLogLocator(ticker.LogLocator):
    """
    LogLocator for Colorbarbar

    This locator is just a `.LogLocator` except the min and max are
    clipped by the norm's min and max (i.e. vmin/vmax from the
    image/pcolor/contour object).  This is necessary so ticks don't
    extrude into the "extend regions".

    """
    def __init__(self, colorbar, *args, **kwargs):
        """
        This ticker needs to know the *colorbar* so that it can access
        its *vmin* and *vmax*.  Otherwise it is the same as
        `~.ticker.LogLocator`.  The ``*args`` and ``**kwargs`` are the
        same as `~.ticker.LogLocator`.
        """
        self._colorbar = colorbar
        super().__init__(*args, **kwargs)

    def tick_values(self, vmin, vmax):
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        vmin = max(vmin, self._colorbar.norm.vmin)
        vmax = min(vmax, self._colorbar.norm.vmax)
        ticks = super().tick_values(vmin, vmax)
        rtol = (np.log10(vmax) - np.log10(vmin)) * 1e-10
        ticks = ticks[(np.log10(ticks) >= np.log10(vmin) - rtol) &
                      (np.log10(ticks) <= np.log10(vmax) + rtol)]
        return ticks


class _ColorbarSpine(mspines.Spine):
    def __init__(self, axes):
        super().__init__(axes, 'colorbar',
                         mpath.Path(np.empty((0, 2)), closed=True))

    def get_window_extent(self, renderer=None):
        # This Spine has no Axis associated with it, and doesn't need to adjust
        # its location, so we can directly get the window extent from the
        # super-super-class.
        return mpatches.Patch.get_window_extent(self, renderer=renderer)

    def set_xy(self, xy):
        self._path = mpath.Path(xy, closed=True)
        self.stale = True

    def draw(self, renderer):
        ret = mpatches.Patch.draw(self, renderer)
        self.stale = False
        return ret


class ColorbarBase:
    r"""
    Draw a colorbar in an existing axes.

    There are only some rare cases in which you would work directly with a
    `.ColorbarBase` as an end-user. Typically, colorbars are used
    with `.ScalarMappable`\s such as an `.AxesImage` generated via
    `~.axes.Axes.imshow`. For these cases you will use `.Colorbar` and
    likely create it via `.pyplot.colorbar` or `.Figure.colorbar`.

    The main application of using a `.ColorbarBase` explicitly is drawing
    colorbars that are not associated with other elements in the figure, e.g.
    when showing a colormap by itself.

    If the *cmap* kwarg is given but *boundaries* and *values* are left as
    None, then the colormap will be displayed on a 0-1 scale. To show the
    under- and over-value colors, specify the *norm* as::

        norm=colors.Normalize(clip=False)

    To show the colors versus index instead of on the 0-1 scale,
    use::

        norm=colors.NoNorm()

    Useful public methods are :meth:`set_label` and :meth:`add_lines`.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` instance in which the colorbar is drawn.
    lines : list
        A list of `.LineCollection` (empty if no lines were drawn).
    dividers : `.LineCollection`
        A LineCollection (empty if *drawedges* is ``False``).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` instance in which the colorbar is drawn.
    cmap : `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        The colormap to use.
    norm : `~matplotlib.colors.Normalize`

    alpha : float
        The colorbar transparency between 0 (transparent) and 1 (opaque).

    values

    boundaries

    orientation : {'vertical', 'horizontal'}

    ticklocation : {'auto', 'left', 'right', 'top', 'bottom'}

    extend : {'neither', 'both', 'min', 'max'}

    spacing : {'uniform', 'proportional'}

    ticks : `~matplotlib.ticker.Locator` or array-like of float

    format : str or `~matplotlib.ticker.Formatter`

    drawedges : bool

    filled : bool

    extendfrac

    extendrec

    label : str
    """

    n_rasterize = 50  # rasterize solids if number of colors >= n_rasterize

    @_api.make_keyword_only("3.3", "cmap")
    def __init__(self, ax, cmap=None,
                 norm=None,
                 alpha=None,
                 values=None,
                 boundaries=None,
                 orientation='vertical',
                 ticklocation='auto',
                 extend=None,
                 spacing='uniform',  # uniform or proportional
                 ticks=None,
                 format=None,
                 drawedges=False,
                 filled=True,
                 extendfrac=None,
                 extendrect=False,
                 label='',
                 ):
        _api.check_isinstance([colors.Colormap, None], cmap=cmap)
        _api.check_in_list(
            ['vertical', 'horizontal'], orientation=orientation)
        _api.check_in_list(
            ['auto', 'left', 'right', 'top', 'bottom'],
            ticklocation=ticklocation)
        _api.check_in_list(
            ['uniform', 'proportional'], spacing=spacing)

        self.ax = ax
        # Bind some methods to the axes to warn users against using them.
        ax.set_xticks = ax.set_yticks = _set_ticks_on_axis_warn
        ax.set(navigate=False)

        if cmap is None:
            cmap = cm.get_cmap()
        if norm is None:
            norm = colors.Normalize()
        if extend is None:
            if hasattr(norm, 'extend'):
                extend = norm.extend
            else:
                extend = 'neither'
        self.alpha = alpha
        self.cmap = cmap
        self.norm = norm
        self.values = values
        self.boundaries = boundaries
        self.extend = extend
        self._inside = _api.check_getitem(
            {'neither': slice(0, None), 'both': slice(1, -1),
             'min': slice(1, None), 'max': slice(0, -1)},
            extend=extend)
        self.spacing = spacing
        self.orientation = orientation
        self.drawedges = drawedges
        self.filled = filled
        self.extendfrac = extendfrac
        self.extendrect = extendrect
        self.solids = None
        self.solids_patches = []
        self.lines = []

        for spine in ax.spines.values():
            spine.set_visible(False)
        self.outline = ax.spines['outline'] = _ColorbarSpine(ax)

        self.patch = mpatches.Polygon(
            np.empty((0, 2)),
            color=mpl.rcParams['axes.facecolor'], linewidth=0.01, zorder=-1)
        ax.add_artist(self.patch)

        self.dividers = collections.LineCollection(
            [],
            colors=[mpl.rcParams['axes.edgecolor']],
            linewidths=[0.5 * mpl.rcParams['axes.linewidth']])
        self.ax.add_collection(self.dividers)

        self.locator = None
        self.formatter = None
        self._manual_tick_data_values = None
        self.__scale = None  # linear, log10 for now.  Hopefully more?

        if ticklocation == 'auto':
            ticklocation = 'bottom' if orientation == 'horizontal' else 'right'
        self.ticklocation = ticklocation

        self.set_label(label)
        self._reset_locator_formatter_scale()

        if np.iterable(ticks):
            self.locator = ticker.FixedLocator(ticks, nbins=len(ticks))
        else:
            self.locator = ticks    # Handle default in _ticker()

        if isinstance(format, str):
            self.formatter = ticker.FormatStrFormatter(format)
        else:
            self.formatter = format  # Assume it is a Formatter or None
        self.draw_all()

    def _extend_lower(self):
        """Return whether the lower limit is open ended."""
        return self.extend in ('both', 'min')

    def _extend_upper(self):
        """Return whether the upper limit is open ended."""
        return self.extend in ('both', 'max')

    def draw_all(self):
        """
        Calculate any free parameters based on the current cmap and norm,
        and do all the drawing.
        """
        self._config_axis()  # Inline it after deprecation elapses.
        # Set self._boundaries and self._values, including extensions.
        self._process_values()
        # Set self.vmin and self.vmax to first and last boundary, excluding
        # extensions.
        self.vmin, self.vmax = self._boundaries[self._inside][[0, -1]]
        # Compute the X/Y mesh.
        X, Y = self._mesh()
        # Extract bounding polygon (the last entry's value (X[0, 1]) doesn't
        # matter, it just matches the CLOSEPOLY code).
        x = np.concatenate([X[[0, 1, -2, -1], 0], X[[-1, -2, 1, 0, 0], 1]])
        y = np.concatenate([Y[[0, 1, -2, -1], 0], Y[[-1, -2, 1, 0, 0], 1]])
        xy = np.column_stack([x, y])
        # Configure axes limits, patch, and outline.
        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0)
        self.ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        self.outline.set_xy(xy)
        self.patch.set_xy(xy)
        self.update_ticks()
        if self.filled:
            self._add_solids(X, Y, self._values[:, np.newaxis])

    @_api.deprecated("3.3")
    def config_axis(self):
        self._config_axis()

    def _config_axis(self):
        """Set up long and short axis."""
        ax = self.ax
        if self.orientation == 'vertical':
            long_axis, short_axis = ax.yaxis, ax.xaxis
            if mpl.rcParams['ytick.minor.visible']:
                self.minorticks_on()
        else:
            long_axis, short_axis = ax.xaxis, ax.yaxis
            if mpl.rcParams['xtick.minor.visible']:
                self.minorticks_on()
        long_axis.set(label_position=self.ticklocation,
                      ticks_position=self.ticklocation)
        short_axis.set_ticks([])
        short_axis.set_ticks([], minor=True)
        self.stale = True

    def _get_ticker_locator_formatter(self):
        """
        Return the ``locator`` and ``formatter`` of the colorbar.

        If they have not been defined (i.e. are *None*), suitable formatter
        and locator instances will be created, attached to the respective
        attributes and returned.
        """
        locator = self.locator
        formatter = self.formatter
        if locator is None:
            if self.boundaries is None:
                if isinstance(self.norm, colors.NoNorm):
                    nv = len(self._values)
                    base = 1 + int(nv / 10)
                    locator = ticker.IndexLocator(base=base, offset=0)
                elif isinstance(self.norm, colors.BoundaryNorm):
                    b = self.norm.boundaries
                    locator = ticker.FixedLocator(b, nbins=10)
                elif isinstance(self.norm, colors.LogNorm):
                    locator = _ColorbarLogLocator(self)
                elif isinstance(self.norm, colors.SymLogNorm):
                    # The subs setting here should be replaced
                    # by logic in the locator.
                    locator = ticker.SymmetricalLogLocator(
                                      subs=np.arange(1, 10),
                                      linthresh=self.norm.linthresh,
                                      base=10)
                else:
                    if mpl.rcParams['_internal.classic_mode']:
                        locator = ticker.MaxNLocator()
                    else:
                        locator = _ColorbarAutoLocator(self)
            else:
                b = self._boundaries[self._inside]
                locator = ticker.FixedLocator(b, nbins=10)

        if formatter is None:
            if isinstance(self.norm, colors.LogNorm):
                formatter = ticker.LogFormatterSciNotation()
            elif isinstance(self.norm, colors.SymLogNorm):
                formatter = ticker.LogFormatterSciNotation(
                                        linthresh=self.norm.linthresh)
            else:
                formatter = ticker.ScalarFormatter()
        else:
            formatter = self.formatter

        self.locator = locator
        self.formatter = formatter
        _log.debug('locator: %r', locator)
        return locator, formatter

    def _use_auto_colorbar_locator(self):
        """
        Return if we should use an adjustable tick locator or a fixed
        one.  (check is used twice so factored out here...)
        """
        contouring = self.boundaries is not None and self.spacing == 'uniform'
        return (type(self.norm) in [colors.Normalize, colors.LogNorm] and
                not contouring)

    def _reset_locator_formatter_scale(self):
        """
        Reset the locator et al to defaults.  Any user-hardcoded changes
        need to be re-entered if this gets called (either at init, or when
        the mappable normal gets changed: Colorbar.update_normal)
        """
        self.locator = None
        self.formatter = None
        if isinstance(self.norm, colors.LogNorm):
            # *both* axes are made log so that determining the
            # mid point is easier.
            self.ax.set_xscale('log')
            self.ax.set_yscale('log')
            self.minorticks_on()
            self.__scale = 'log'
        else:
            self.ax.set_xscale('linear')
            self.ax.set_yscale('linear')
            if type(self.norm) is colors.Normalize:
                self.__scale = 'linear'
            else:
                self.__scale = 'manual'

    def update_ticks(self):
        """
        Force the update of the ticks and ticklabels. This must be
        called whenever the tick locator and/or tick formatter changes.
        """
        ax = self.ax
        # Get the locator and formatter; defaults to self.locator if not None.
        locator, formatter = self._get_ticker_locator_formatter()
        long_axis = ax.yaxis if self.orientation == 'vertical' else ax.xaxis
        if self._use_auto_colorbar_locator():
            _log.debug('Using auto colorbar locator %r on colorbar', locator)
            long_axis.set_major_locator(locator)
            long_axis.set_major_formatter(formatter)
        else:
            _log.debug('Using fixed locator on colorbar')
            ticks, ticklabels, offset_string = self._ticker(locator, formatter)
            long_axis.set_ticks(ticks)
            long_axis.set_ticklabels(ticklabels)
            long_axis.get_major_formatter().set_offset_string(offset_string)

    def set_ticks(self, ticks, update_ticks=True):
        """
        Set tick locations.

        Parameters
        ----------
        ticks : array-like or `~matplotlib.ticker.Locator` or None
            The tick positions can be hard-coded by an array of values; or
            they can be defined by a `.Locator`. Setting to *None* reverts
            to using a default locator.

        update_ticks : bool, default: True
            If True, tick locations are updated immediately.  If False, the
            user has to call `update_ticks` later to update the ticks.

        """
        if np.iterable(ticks):
            self.locator = ticker.FixedLocator(ticks, nbins=len(ticks))
        else:
            self.locator = ticks

        if update_ticks:
            self.update_ticks()
        self.stale = True

    def get_ticks(self, minor=False):
        """Return the x ticks as a list of locations."""
        if self._manual_tick_data_values is None:
            ax = self.ax
            long_axis = (
                ax.yaxis if self.orientation == 'vertical' else ax.xaxis)
            return long_axis.get_majorticklocs()
        else:
            # We made the axes manually, the old way, and the ylim is 0-1,
            # so the majorticklocs are in those units, not data units.
            return self._manual_tick_data_values

    def set_ticklabels(self, ticklabels, update_ticks=True):
        """
        Set tick labels.

        Tick labels are updated immediately unless *update_ticks* is *False*,
        in which case one should call `.update_ticks` explicitly.
        """
        if isinstance(self.locator, ticker.FixedLocator):
            self.formatter = ticker.FixedFormatter(ticklabels)
            if update_ticks:
                self.update_ticks()
        else:
            _api.warn_external("set_ticks() must have been called.")
        self.stale = True

    def minorticks_on(self):
        """
        Turn the minor ticks of the colorbar on without extruding
        into the "extend regions".
        """
        ax = self.ax
        long_axis = ax.yaxis if self.orientation == 'vertical' else ax.xaxis

        if long_axis.get_scale() == 'log':
            long_axis.set_minor_locator(_ColorbarLogLocator(self, base=10.,
                                                            subs='auto'))
            long_axis.set_minor_formatter(ticker.LogFormatterSciNotation())
        else:
            long_axis.set_minor_locator(_ColorbarAutoMinorLocator(self))

    def minorticks_off(self):
        """Turn the minor ticks of the colorbar off."""
        ax = self.ax
        long_axis = ax.yaxis if self.orientation == 'vertical' else ax.xaxis
        long_axis.set_minor_locator(ticker.NullLocator())

    def set_label(self, label, *, loc=None, **kwargs):
        """
        Add a label to the long axis of the colorbar.

        Parameters
        ----------
        label : str
            The label text.
        loc : str, optional
            The location of the label.

            - For horizontal orientation one of {'left', 'center', 'right'}
            - For vertical orientation one of {'bottom', 'center', 'top'}

            Defaults to :rc:`xaxis.labellocation` or :rc:`yaxis.labellocation`
            depending on the orientation.
        **kwargs
            Keyword arguments are passed to `~.Axes.set_xlabel` /
            `~.Axes.set_ylabel`.
            Supported keywords are *labelpad* and `.Text` properties.
        """
        if self.orientation == "vertical":
            self.ax.set_ylabel(label, loc=loc, **kwargs)
        else:
            self.ax.set_xlabel(label, loc=loc, **kwargs)
        self.stale = True

    def _add_solids(self, X, Y, C):
        """Draw the colors; optionally add separators."""
        # Cleanup previously set artists.
        if self.solids is not None:
            self.solids.remove()
        for solid in self.solids_patches:
            solid.remove()
        # Add new artist(s), based on mappable type.  Use individual patches if
        # hatching is needed, pcolormesh otherwise.
        mappable = getattr(self, 'mappable', None)
        if (isinstance(mappable, contour.ContourSet)
                and any(hatch is not None for hatch in mappable.hatches)):
            self._add_solids_patches(X, Y, C, mappable)
        else:
            self._add_solids_pcolormesh(X, Y, C)
        self.dividers.set_segments(
            np.dstack([X, Y])[1:-1] if self.drawedges else [])

    def _add_solids_pcolormesh(self, X, Y, C):
        _log.debug('Setting pcolormesh')
        if C.shape[0] == Y.shape[0]:
            # trim the last one to be compatible with old behavior.
            C = C[:-1]
        self.solids = self.ax.pcolormesh(
            X, Y, C, cmap=self.cmap, norm=self.norm, alpha=self.alpha,
            edgecolors='none', shading='flat')
        if not self.drawedges:
            if len(self._y) >= self.n_rasterize:
                self.solids.set_rasterized(True)

    def _add_solids_patches(self, X, Y, C, mappable):
        hatches = mappable.hatches * len(C)  # Have enough hatches.
        patches = []
        for i in range(len(X) - 1):
            xy = np.array([[X[i, 0], Y[i, 0]],
                           [X[i, 1], Y[i, 0]],
                           [X[i + 1, 1], Y[i + 1, 0]],
                           [X[i + 1, 0], Y[i + 1, 1]]])
            patch = mpatches.PathPatch(mpath.Path(xy),
                                       facecolor=self.cmap(self.norm(C[i][0])),
                                       hatch=hatches[i], linewidth=0,
                                       antialiased=False, alpha=self.alpha)
            self.ax.add_patch(patch)
            patches.append(patch)
        self.solids_patches = patches

    def add_lines(self, levels, colors, linewidths, erase=True):
        """
        Draw lines on the colorbar.

        The lines are appended to the list :attr:`lines`.

        Parameters
        ----------
        levels : array-like
            The positions of the lines.
        colors : color or list of colors
            Either a single color applying to all lines or one color value for
            each line.
        linewidths : float or array-like
            Either a single linewidth applying to all lines or one linewidth
            for each line.
        erase : bool, default: True
            Whether to remove any previously added lines.
        """
        y = self._locate(levels)
        rtol = (self._y[-1] - self._y[0]) * 1e-10
        igood = (y < self._y[-1] + rtol) & (y > self._y[0] - rtol)
        y = y[igood]
        if np.iterable(colors):
            colors = np.asarray(colors)[igood]
        if np.iterable(linewidths):
            linewidths = np.asarray(linewidths)[igood]
        X, Y = np.meshgrid([self._y[0], self._y[-1]], y)
        if self.orientation == 'vertical':
            xy = np.stack([X, Y], axis=-1)
        else:
            xy = np.stack([Y, X], axis=-1)
        col = collections.LineCollection(xy, linewidths=linewidths)

        if erase and self.lines:
            for lc in self.lines:
                lc.remove()
            self.lines = []
        self.lines.append(col)
        col.set_color(colors)
        self.ax.add_collection(col)
        self.stale = True

    def _ticker(self, locator, formatter):
        """
        Return the sequence of ticks (colorbar data locations),
        ticklabels (strings), and the corresponding offset string.
        """
        if isinstance(self.norm, colors.NoNorm) and self.boundaries is None:
            intv = self._values[0], self._values[-1]
        else:
            intv = self.vmin, self.vmax
        locator.create_dummy_axis(minpos=intv[0])
        formatter.create_dummy_axis(minpos=intv[0])
        locator.set_view_interval(*intv)
        locator.set_data_interval(*intv)
        formatter.set_view_interval(*intv)
        formatter.set_data_interval(*intv)

        b = np.array(locator())
        if isinstance(locator, ticker.LogLocator):
            eps = 1e-10
            b = b[(b <= intv[1] * (1 + eps)) & (b >= intv[0] * (1 - eps))]
        else:
            eps = (intv[1] - intv[0]) * 1e-10
            b = b[(b <= intv[1] + eps) & (b >= intv[0] - eps)]
        self._manual_tick_data_values = b
        ticks = self._locate(b)
        ticklabels = formatter.format_ticks(b)
        offset_string = formatter.get_offset()
        return ticks, ticklabels, offset_string

    def _process_values(self, b=None):
        """
        Set the :attr:`_boundaries` and :attr:`_values` attributes
        based on the input boundaries and values.  Input boundaries
        can be *self.boundaries* or the argument *b*.
        """
        if b is None:
            b = self.boundaries
        if b is not None:
            self._boundaries = np.asarray(b, dtype=float)
            if self.values is None:
                self._values = 0.5 * (self._boundaries[:-1]
                                      + self._boundaries[1:])
                if isinstance(self.norm, colors.NoNorm):
                    self._values = (self._values + 0.00001).astype(np.int16)
            else:
                self._values = np.array(self.values)
            return
        if self.values is not None:
            self._values = np.array(self.values)
            if self.boundaries is None:
                b = np.zeros(len(self.values) + 1)
                b[1:-1] = 0.5 * (self._values[:-1] + self._values[1:])
                b[0] = 2.0 * b[1] - b[2]
                b[-1] = 2.0 * b[-2] - b[-3]
                self._boundaries = b
                return
            self._boundaries = np.array(self.boundaries)
            return
        # Neither boundaries nor values are specified;
        # make reasonable ones based on cmap and norm.
        if isinstance(self.norm, colors.NoNorm):
            b = self._uniform_y(self.cmap.N + 1) * self.cmap.N - 0.5
            v = np.zeros(len(b) - 1, dtype=np.int16)
            v[self._inside] = np.arange(self.cmap.N, dtype=np.int16)
            if self._extend_lower():
                v[0] = -1
            if self._extend_upper():
                v[-1] = self.cmap.N
            self._boundaries = b
            self._values = v
            return
        elif isinstance(self.norm, colors.BoundaryNorm):
            b = list(self.norm.boundaries)
            if self._extend_lower():
                b = [b[0] - 1] + b
            if self._extend_upper():
                b = b + [b[-1] + 1]
            b = np.array(b)
            v = np.zeros(len(b) - 1)
            bi = self.norm.boundaries
            v[self._inside] = 0.5 * (bi[:-1] + bi[1:])
            if self._extend_lower():
                v[0] = b[0] - 1
            if self._extend_upper():
                v[-1] = b[-1] + 1
            self._boundaries = b
            self._values = v
            return
        else:
            if not self.norm.scaled():
                self.norm.vmin = 0
                self.norm.vmax = 1

            self.norm.vmin, self.norm.vmax = mtransforms.nonsingular(
                self.norm.vmin,
                self.norm.vmax,
                expander=0.1)

            b = self.norm.inverse(self._uniform_y(self.cmap.N + 1))

            if isinstance(self.norm, (colors.PowerNorm, colors.LogNorm)):
                # If using a lognorm or powernorm, ensure extensions don't
                # go negative
                if self._extend_lower():
                    b[0] = 0.9 * b[0]
                if self._extend_upper():
                    b[-1] = 1.1 * b[-1]
            else:
                if self._extend_lower():
                    b[0] = b[0] - 1
                if self._extend_upper():
                    b[-1] = b[-1] + 1
        self._process_values(b)

    def _get_extension_lengths(self, frac, automin, automax, default=0.05):
        """
        Return the lengths of colorbar extensions.

        This is a helper method for _uniform_y and _proportional_y.
        """
        # Set the default value.
        extendlength = np.array([default, default])
        if isinstance(frac, str):
            _api.check_in_list(['auto'], extendfrac=frac.lower())
            # Use the provided values when 'auto' is required.
            extendlength[:] = [automin, automax]
        elif frac is not None:
            try:
                # Try to set min and max extension fractions directly.
                extendlength[:] = frac
                # If frac is a sequence containing None then NaN may
                # be encountered. This is an error.
                if np.isnan(extendlength).any():
                    raise ValueError()
            except (TypeError, ValueError) as err:
                # Raise an error on encountering an invalid value for frac.
                raise ValueError('invalid value for extendfrac') from err
        return extendlength

    def _uniform_y(self, N):
        """
        Return colorbar data coordinates for *N* uniformly
        spaced boundaries, plus ends if required.
        """
        if self.extend == 'neither':
            y = np.linspace(0, 1, N)
        else:
            automin = automax = 1. / (N - 1.)
            extendlength = self._get_extension_lengths(self.extendfrac,
                                                       automin, automax,
                                                       default=0.05)
            if self.extend == 'both':
                y = np.zeros(N + 2, 'd')
                y[0] = 0. - extendlength[0]
                y[-1] = 1. + extendlength[1]
            elif self.extend == 'min':
                y = np.zeros(N + 1, 'd')
                y[0] = 0. - extendlength[0]
            else:
                y = np.zeros(N + 1, 'd')
                y[-1] = 1. + extendlength[1]
            y[self._inside] = np.linspace(0, 1, N)
        return y

    def _proportional_y(self):
        """
        Return colorbar data coordinates for the boundaries of
        a proportional colorbar.
        """
        if isinstance(self.norm, colors.BoundaryNorm):
            y = (self._boundaries - self._boundaries[0])
            y = y / (self._boundaries[-1] - self._boundaries[0])
        else:
            y = self.norm(self._boundaries.copy())
            y = np.ma.filled(y, np.nan)
        if self.extend == 'min':
            # Exclude leftmost interval of y.
            clen = y[-1] - y[1]
            automin = (y[2] - y[1]) / clen
            automax = (y[-1] - y[-2]) / clen
        elif self.extend == 'max':
            # Exclude rightmost interval in y.
            clen = y[-2] - y[0]
            automin = (y[1] - y[0]) / clen
            automax = (y[-2] - y[-3]) / clen
        elif self.extend == 'both':
            # Exclude leftmost and rightmost intervals in y.
            clen = y[-2] - y[1]
            automin = (y[2] - y[1]) / clen
            automax = (y[-2] - y[-3]) / clen
        if self.extend in ('both', 'min', 'max'):
            extendlength = self._get_extension_lengths(self.extendfrac,
                                                       automin, automax,
                                                       default=0.05)
        if self.extend in ('both', 'min'):
            y[0] = 0. - extendlength[0]
        if self.extend in ('both', 'max'):
            y[-1] = 1. + extendlength[1]
        yi = y[self._inside]
        norm = colors.Normalize(yi[0], yi[-1])
        y[self._inside] = np.ma.filled(norm(yi), np.nan)
        return y

    def _mesh(self):
        """
        Return the coordinate arrays for the colorbar pcolormesh/patches.

        These are scaled between vmin and vmax, and already handle colorbar
        orientation.
        """
        # copy the norm and change the vmin and vmax to the vmin and
        # vmax of the colorbar, not the norm.  This allows the situation
        # where the colormap has a narrower range than the colorbar, to
        # accommodate extra contours:
        norm = copy.copy(self.norm)
        norm.vmin = self.vmin
        norm.vmax = self.vmax
        x = np.array([0.0, 1.0])
        if self.spacing == 'uniform':
            n_boundaries_no_extensions = len(self._boundaries[self._inside])
            y = self._uniform_y(n_boundaries_no_extensions)
        else:
            y = self._proportional_y()
        xmid = np.array([0.5])
        if self.__scale != 'manual':
            y = norm.inverse(y)
            x = norm.inverse(x)
            xmid = norm.inverse(xmid)
        else:
            # if a norm doesn't have a named scale, or
            # we are not using a norm
            dv = self.vmax - self.vmin
            x = x * dv + self.vmin
            y = y * dv + self.vmin
            xmid = xmid * dv + self.vmin
        self._y = y
        X, Y = np.meshgrid(x, y)
        if self._extend_lower() and not self.extendrect:
            X[0, :] = xmid
        if self._extend_upper() and not self.extendrect:
            X[-1, :] = xmid
        return (X, Y) if self.orientation == 'vertical' else (Y, X)

    def _locate(self, x):
        """
        Given a set of color data values, return their
        corresponding colorbar data coordinates.
        """
        if isinstance(self.norm, (colors.NoNorm, colors.BoundaryNorm)):
            b = self._boundaries
            xn = x
        else:
            # Do calculations using normalized coordinates so
            # as to make the interpolation more accurate.
            b = self.norm(self._boundaries, clip=False).filled()
            xn = self.norm(x, clip=False).filled()

        bunique = b
        yunique = self._y
        # trim extra b values at beginning and end if they are
        # not unique.  These are here for extended colorbars, and are not
        # wanted for the interpolation.
        if b[0] == b[1]:
            bunique = bunique[1:]
            yunique = yunique[1:]
        if b[-1] == b[-2]:
            bunique = bunique[:-1]
            yunique = yunique[:-1]

        z = np.interp(xn, bunique, yunique)
        return z

    def set_alpha(self, alpha):
        """Set the transparency between 0 (transparent) and 1 (opaque)."""
        self.alpha = alpha

    def remove(self):
        """Remove this colorbar from the figure."""
        self.ax.remove()


def _add_disjoint_kwargs(d, **kwargs):
    """
    Update dict *d* with entries in *kwargs*, which must be absent from *d*.
    """
    for k, v in kwargs.items():
        if k in d:
            _api.warn_deprecated(
                "3.3", message=f"The {k!r} parameter to Colorbar has no "
                "effect because it is overridden by the mappable; it is "
                "deprecated since %(since)s and will be removed %(removal)s.")
        d[k] = v


class Colorbar(ColorbarBase):
    """
    This class connects a `ColorbarBase` to a `~.cm.ScalarMappable`
    such as an `~.image.AxesImage` generated via `~.axes.Axes.imshow`.

    .. note::
        This class is not intended to be instantiated directly; instead, use
        `.Figure.colorbar` or `.pyplot.colorbar` to create a colorbar.
    """

    def __init__(self, ax, mappable, **kwargs):
        # Ensure the given mappable's norm has appropriate vmin and vmax set
        # even if mappable.draw has not yet been called.
        if mappable.get_array() is not None:
            mappable.autoscale_None()

        self.mappable = mappable
        _add_disjoint_kwargs(kwargs, cmap=mappable.cmap, norm=mappable.norm)

        if isinstance(mappable, contour.ContourSet):
            cs = mappable
            _add_disjoint_kwargs(
                kwargs,
                alpha=cs.get_alpha(),
                boundaries=cs._levels,
                values=cs.cvalues,
                extend=cs.extend,
                filled=cs.filled,
            )
            kwargs.setdefault(
                'ticks', ticker.FixedLocator(cs.levels, nbins=10))
            super().__init__(ax, **kwargs)
            if not cs.filled:
                self.add_lines(cs)
        else:
            if getattr(mappable.cmap, 'colorbar_extend', False) is not False:
                kwargs.setdefault('extend', mappable.cmap.colorbar_extend)
            if isinstance(mappable, martist.Artist):
                _add_disjoint_kwargs(kwargs, alpha=mappable.get_alpha())
            super().__init__(ax, **kwargs)

        mappable.colorbar = self
        mappable.colorbar_cid = mappable.callbacksSM.connect(
            'changed', self.update_normal)

    @_api.deprecated("3.3", alternative="update_normal")
    def on_mappable_changed(self, mappable):
        """
        Update this colorbar to match the mappable's properties.

        Typically this is automatically registered as an event handler
        by :func:`colorbar_factory` and should not be called manually.
        """
        _log.debug('colorbar mappable changed')
        self.update_normal(mappable)

    def add_lines(self, CS, erase=True):
        """
        Add the lines from a non-filled `~.contour.ContourSet` to the colorbar.

        Parameters
        ----------
        CS : `~.contour.ContourSet`
            The line positions are taken from the ContourSet levels. The
            ContourSet must not be filled.
        erase : bool, default: True
            Whether to remove any previously added lines.
        """
        if not isinstance(CS, contour.ContourSet) or CS.filled:
            raise ValueError('add_lines is only for a ContourSet of lines')
        tcolors = [c[0] for c in CS.tcolors]
        tlinewidths = [t[0] for t in CS.tlinewidths]
        # Wishlist: Make colorbar lines auto-follow changes in contour lines.
        super().add_lines(CS.levels, tcolors, tlinewidths, erase=erase)

    def update_normal(self, mappable):
        """
        Update solid patches, lines, etc.

        This is meant to be called when the norm of the image or contour plot
        to which this colorbar belongs changes.

        If the norm on the mappable is different than before, this resets the
        locator and formatter for the axis, so if these have been customized,
        they will need to be customized again.  However, if the norm only
        changes values of *vmin*, *vmax* or *cmap* then the old formatter
        and locator will be preserved.
        """
        _log.debug('colorbar update normal %r %r', mappable.norm, self.norm)
        self.mappable = mappable
        self.set_alpha(mappable.get_alpha())
        self.cmap = mappable.cmap
        if mappable.norm != self.norm:
            self.norm = mappable.norm
            self._reset_locator_formatter_scale()

        self.draw_all()
        if isinstance(self.mappable, contour.ContourSet):
            CS = self.mappable
            if not CS.filled:
                self.add_lines(CS)
        self.stale = True

    @_api.deprecated("3.3", alternative="update_normal")
    def update_bruteforce(self, mappable):
        """
        Destroy and rebuild the colorbar.  This is
        intended to become obsolete, and will probably be
        deprecated and then removed.  It is not called when
        the pyplot.colorbar function or the Figure.colorbar
        method are used to create the colorbar.
        """
        # We are using an ugly brute-force method: clearing and
        # redrawing the whole thing.  The problem is that if any
        # properties have been changed by methods other than the
        # colorbar methods, those changes will be lost.
        self.ax.cla()
        self.locator = None
        self.formatter = None

        # clearing the axes will delete outline, patch, solids, and lines:
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.outline = self.ax.spines['outline'] = _ColorbarSpine(self.ax)
        self.patch = mpatches.Polygon(
            np.empty((0, 2)),
            color=mpl.rcParams['axes.facecolor'], linewidth=0.01, zorder=-1)
        self.ax.add_artist(self.patch)
        self.solids = None
        self.lines = []
        self.update_normal(mappable)
        self.draw_all()
        if isinstance(self.mappable, contour.ContourSet):
            CS = self.mappable
            if not CS.filled:
                self.add_lines(CS)
            #if self.lines is not None:
            #    tcolors = [c[0] for c in CS.tcolors]
            #    self.lines.set_color(tcolors)
        #Fixme? Recalculate boundaries, ticks if vmin, vmax have changed.
        #Fixme: Some refactoring may be needed; we should not
        # be recalculating everything if there was a simple alpha
        # change.

    def remove(self):
        """
        Remove this colorbar from the figure.

        If the colorbar was created with ``use_gridspec=True`` the previous
        gridspec is restored.
        """
        super().remove()
        self.mappable.callbacksSM.disconnect(self.mappable.colorbar_cid)
        self.mappable.colorbar = None
        self.mappable.colorbar_cid = None

        try:
            ax = self.mappable.axes
        except AttributeError:
            return

        try:
            gs = ax.get_subplotspec().get_gridspec()
            subplotspec = gs.get_topmost_subplotspec()
        except AttributeError:
            # use_gridspec was False
            pos = ax.get_position(original=True)
            ax._set_position(pos)
        else:
            # use_gridspec was True
            ax.set_subplotspec(subplotspec)


def _normalize_location_orientation(location, orientation):
    if location is None:
        location = _api.check_getitem(
            {None: "right", "vertical": "right", "horizontal": "bottom"},
            orientation=orientation)
    loc_settings = _api.check_getitem({
        "left":   {"location": "left", "orientation": "vertical",
                   "anchor": (1.0, 0.5), "panchor": (0.0, 0.5), "pad": 0.10},
        "right":  {"location": "right", "orientation": "vertical",
                   "anchor": (0.0, 0.5), "panchor": (1.0, 0.5), "pad": 0.05},
        "top":    {"location": "top", "orientation": "horizontal",
                   "anchor": (0.5, 0.0), "panchor": (0.5, 1.0), "pad": 0.05},
        "bottom": {"location": "bottom", "orientation": "horizontal",
                   "anchor": (0.5, 1.0), "panchor": (0.5, 0.0), "pad": 0.15},
    }, location=location)
    if orientation is not None and orientation != loc_settings["orientation"]:
        # Allow the user to pass both if they are consistent.
        raise TypeError("location and orientation are mutually exclusive")
    return loc_settings


@docstring.Substitution(_make_axes_param_doc, _make_axes_other_param_doc)
def make_axes(parents, location=None, orientation=None, fraction=0.15,
              shrink=1.0, aspect=20, **kw):
    """
    Create an `~.axes.Axes` suitable for a colorbar.

    The axes is placed in the figure of the *parents* axes, by resizing and
    repositioning *parents*.

    Parameters
    ----------
    parents : `~.axes.Axes` or list of `~.axes.Axes`
        The Axes to use as parents for placing the colorbar.
    %s

    Returns
    -------
    cax : `~.axes.Axes`
        The child axes.
    kw : dict
        The reduced keyword dictionary to be passed when creating the colorbar
        instance.

    Other Parameters
    ----------------
    %s
    """
    loc_settings = _normalize_location_orientation(location, orientation)
    # put appropriate values into the kw dict for passing back to
    # the Colorbar class
    kw['orientation'] = loc_settings['orientation']
    location = kw['ticklocation'] = loc_settings['location']

    anchor = kw.pop('anchor', loc_settings['anchor'])
    parent_anchor = kw.pop('panchor', loc_settings['panchor'])

    parents_iterable = np.iterable(parents)
    # turn parents into a list if it is not already. We do this w/ np
    # because `plt.subplots` can return an ndarray and is natural to
    # pass to `colorbar`.
    parents = np.atleast_1d(parents).ravel()
    fig = parents[0].get_figure()

    pad0 = 0.05 if fig.get_constrained_layout() else loc_settings['pad']
    pad = kw.pop('pad', pad0)

    if not all(fig is ax.get_figure() for ax in parents):
        raise ValueError('Unable to create a colorbar axes as not all '
                         'parents share the same figure.')

    # take a bounding box around all of the given axes
    parents_bbox = mtransforms.Bbox.union(
        [ax.get_position(original=True).frozen() for ax in parents])

    pb = parents_bbox
    if location in ('left', 'right'):
        if location == 'left':
            pbcb, _, pb1 = pb.splitx(fraction, fraction + pad)
        else:
            pb1, _, pbcb = pb.splitx(1 - fraction - pad, 1 - fraction)
        pbcb = pbcb.shrunk(1.0, shrink).anchored(anchor, pbcb)
    else:
        if location == 'bottom':
            pbcb, _, pb1 = pb.splity(fraction, fraction + pad)
        else:
            pb1, _, pbcb = pb.splity(1 - fraction - pad, 1 - fraction)
        pbcb = pbcb.shrunk(shrink, 1.0).anchored(anchor, pbcb)

        # define the aspect ratio in terms of y's per x rather than x's per y
        aspect = 1.0 / aspect

    # define a transform which takes us from old axes coordinates to
    # new axes coordinates
    shrinking_trans = mtransforms.BboxTransform(parents_bbox, pb1)

    # transform each of the axes in parents using the new transform
    for ax in parents:
        new_posn = shrinking_trans.transform(ax.get_position(original=True))
        new_posn = mtransforms.Bbox(new_posn)
        ax._set_position(new_posn)
        if parent_anchor is not False:
            ax.set_anchor(parent_anchor)

    cax = fig.add_axes(pbcb, label="<colorbar>")
    for a in parents:
        # tell the parent it has a colorbar
        a._colorbars += [cax]
    cax._colorbar_info = dict(
        location=location,
        parents=parents,
        shrink=shrink,
        anchor=anchor,
        panchor=parent_anchor,
        fraction=fraction,
        aspect=aspect,
        pad=pad)
    # and we need to set the aspect ratio by hand...
    cax.set_aspect(aspect, anchor=anchor, adjustable='box')

    return cax, kw


@docstring.Substitution(_make_axes_param_doc, _make_axes_other_param_doc)
def make_axes_gridspec(parent, *, location=None, orientation=None,
                       fraction=0.15, shrink=1.0, aspect=20, **kw):
    """
    Create a `~.SubplotBase` suitable for a colorbar.

    The axes is placed in the figure of the *parent* axes, by resizing and
    repositioning *parent*.

    This function is similar to `.make_axes`. Primary differences are

    - `.make_axes_gridspec` should only be used with a `.SubplotBase` parent.

    - `.make_axes` creates an `~.axes.Axes`; `.make_axes_gridspec` creates a
      `.SubplotBase`.

    - `.make_axes` updates the position of the parent.  `.make_axes_gridspec`
      replaces the ``grid_spec`` attribute of the parent with a new one.

    While this function is meant to be compatible with `.make_axes`,
    there could be some minor differences.

    Parameters
    ----------
    parent : `~.axes.Axes`
        The Axes to use as parent for placing the colorbar.
    %s

    Returns
    -------
    cax : `~.axes.SubplotBase`
        The child axes.
    kw : dict
        The reduced keyword dictionary to be passed when creating the colorbar
        instance.

    Other Parameters
    ----------------
    %s
    """

    loc_settings = _normalize_location_orientation(location, orientation)
    kw['orientation'] = loc_settings['orientation']
    location = kw['ticklocation'] = loc_settings['location']

    anchor = kw.pop('anchor', loc_settings['anchor'])
    panchor = kw.pop('panchor', loc_settings['panchor'])
    pad = kw.pop('pad', loc_settings["pad"])
    wh_space = 2 * pad / (1 - pad)

    if location in ('left', 'right'):
        # for shrinking
        height_ratios = [
                (1-anchor[1])*(1-shrink), shrink, anchor[1]*(1-shrink)]

        if location == 'left':
            gs = parent.get_subplotspec().subgridspec(
                    1, 2, wspace=wh_space,
                    width_ratios=[fraction, 1-fraction-pad])
            ss_main = gs[1]
            ss_cb = gs[0].subgridspec(
                    3, 1, hspace=0, height_ratios=height_ratios)[1]
        else:
            gs = parent.get_subplotspec().subgridspec(
                    1, 2, wspace=wh_space,
                    width_ratios=[1-fraction-pad, fraction])
            ss_main = gs[0]
            ss_cb = gs[1].subgridspec(
                    3, 1, hspace=0, height_ratios=height_ratios)[1]
    else:
        # for shrinking
        width_ratios = [
                anchor[0]*(1-shrink), shrink, (1-anchor[0])*(1-shrink)]

        if location == 'bottom':
            gs = parent.get_subplotspec().subgridspec(
                    2, 1, hspace=wh_space,
                    height_ratios=[1-fraction-pad, fraction])
            ss_main = gs[0]
            ss_cb = gs[1].subgridspec(
                    1, 3, wspace=0, width_ratios=width_ratios)[1]
            aspect = 1 / aspect
        else:
            gs = parent.get_subplotspec().subgridspec(
                    2, 1, hspace=wh_space,
                    height_ratios=[fraction, 1-fraction-pad])
            ss_main = gs[1]
            ss_cb = gs[0].subgridspec(
                    1, 3, wspace=0, width_ratios=width_ratios)[1]
            aspect = 1 / aspect

    parent.set_subplotspec(ss_main)
    parent.set_anchor(loc_settings["panchor"])

    fig = parent.get_figure()
    cax = fig.add_subplot(ss_cb, label="<colorbar>")
    cax.set_aspect(aspect, anchor=loc_settings["anchor"], adjustable='box')
    return cax, kw


@_api.deprecated("3.4", alternative="Colorbar")
class ColorbarPatch(Colorbar):
    pass


@_api.deprecated("3.4", alternative="Colorbar")
def colorbar_factory(cax, mappable, **kwargs):
    """
    Create a colorbar on the given axes for the given mappable.

    .. note::
        This is a low-level function to turn an existing axes into a colorbar
        axes.  Typically, you'll want to use `~.Figure.colorbar` instead, which
        automatically handles creation and placement of a suitable axes as
        well.

    Parameters
    ----------
    cax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` to turn into a colorbar.
    mappable : `~matplotlib.cm.ScalarMappable`
        The mappable to be described by the colorbar.
    **kwargs
        Keyword arguments are passed to the respective colorbar class.

    Returns
    -------
    `.Colorbar`
        The created colorbar instance.
    """
    return Colorbar(cax, mappable, **kwargs)
