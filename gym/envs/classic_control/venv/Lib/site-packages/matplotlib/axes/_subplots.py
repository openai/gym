import functools

from matplotlib import _api, docstring
import matplotlib.artist as martist
from matplotlib.axes._axes import Axes
from matplotlib.gridspec import GridSpec, SubplotSpec


class SubplotBase:
    """
    Base class for subplots, which are :class:`Axes` instances with
    additional methods to facilitate generating and manipulating a set
    of :class:`Axes` within a figure.
    """

    def __init__(self, fig, *args, **kwargs):
        """
        Parameters
        ----------
        fig : `matplotlib.figure.Figure`

        *args : tuple (*nrows*, *ncols*, *index*) or int
            The array of subplots in the figure has dimensions ``(nrows,
            ncols)``, and *index* is the index of the subplot being created.
            *index* starts at 1 in the upper left corner and increases to the
            right.

            If *nrows*, *ncols*, and *index* are all single digit numbers, then
            *args* can be passed as a single 3-digit number (e.g. 234 for
            (2, 3, 4)).

        **kwargs
            Keyword arguments are passed to the Axes (sub)class constructor.
        """
        # _axes_class is set in the subplot_class_factory
        self._axes_class.__init__(self, fig, [0, 0, 1, 1], **kwargs)
        # This will also update the axes position.
        self.set_subplotspec(SubplotSpec._from_subplot_args(fig, args))

    def __reduce__(self):
        # get the first axes class which does not inherit from a subplotbase
        axes_class = next(
            c for c in type(self).__mro__
            if issubclass(c, Axes) and not issubclass(c, SubplotBase))
        return (_picklable_subplot_class_constructor,
                (axes_class,),
                self.__getstate__())

    @_api.deprecated(
        "3.4", alternative="get_subplotspec",
        addendum="(get_subplotspec returns a SubplotSpec instance.)")
    def get_geometry(self):
        """Get the subplot geometry, e.g., (2, 2, 3)."""
        rows, cols, num1, num2 = self.get_subplotspec().get_geometry()
        return rows, cols, num1 + 1  # for compatibility

    @_api.deprecated("3.4", alternative="set_subplotspec")
    def change_geometry(self, numrows, numcols, num):
        """Change subplot geometry, e.g., from (1, 1, 1) to (2, 2, 3)."""
        self._subplotspec = GridSpec(numrows, numcols,
                                     figure=self.figure)[num - 1]
        self.update_params()
        self.set_position(self.figbox)

    def get_subplotspec(self):
        """Return the `.SubplotSpec` instance associated with the subplot."""
        return self._subplotspec

    def set_subplotspec(self, subplotspec):
        """Set the `.SubplotSpec`. instance associated with the subplot."""
        self._subplotspec = subplotspec
        self._set_position(subplotspec.get_position(self.figure))

    def get_gridspec(self):
        """Return the `.GridSpec` instance associated with the subplot."""
        return self._subplotspec.get_gridspec()

    @_api.deprecated(
        "3.4", alternative="get_subplotspec().get_position(self.figure)")
    @property
    def figbox(self):
        return self.get_subplotspec().get_position(self.figure)

    @_api.deprecated("3.4", alternative="get_gridspec().nrows")
    @property
    def numRows(self):
        return self.get_gridspec().nrows

    @_api.deprecated("3.4", alternative="get_gridspec().ncols")
    @property
    def numCols(self):
        return self.get_gridspec().ncols

    @_api.deprecated("3.4")
    def update_params(self):
        """Update the subplot position from ``self.figure.subplotpars``."""
        # Now a no-op, as figbox/numRows/numCols are (deprecated) auto-updating
        # properties.

    @_api.deprecated("3.4", alternative="ax.get_subplotspec().is_first_row()")
    def is_first_row(self):
        return self.get_subplotspec().rowspan.start == 0

    @_api.deprecated("3.4", alternative="ax.get_subplotspec().is_last_row()")
    def is_last_row(self):
        return self.get_subplotspec().rowspan.stop == self.get_gridspec().nrows

    @_api.deprecated("3.4", alternative="ax.get_subplotspec().is_first_col()")
    def is_first_col(self):
        return self.get_subplotspec().colspan.start == 0

    @_api.deprecated("3.4", alternative="ax.get_subplotspec().is_last_col()")
    def is_last_col(self):
        return self.get_subplotspec().colspan.stop == self.get_gridspec().ncols

    def label_outer(self):
        """
        Only show "outer" labels and tick labels.

        x-labels are only kept for subplots on the last row; y-labels only for
        subplots on the first column.
        """
        ss = self.get_subplotspec()
        lastrow = ss.is_last_row()
        firstcol = ss.is_first_col()
        if not lastrow:
            for label in self.get_xticklabels(which="both"):
                label.set_visible(False)
            self.xaxis.get_offset_text().set_visible(False)
            self.set_xlabel("")
        if not firstcol:
            for label in self.get_yticklabels(which="both"):
                label.set_visible(False)
            self.yaxis.get_offset_text().set_visible(False)
            self.set_ylabel("")

    def _make_twin_axes(self, *args, **kwargs):
        """Make a twinx axes of self. This is used for twinx and twiny."""
        if 'sharex' in kwargs and 'sharey' in kwargs:
            # The following line is added in v2.2 to avoid breaking Seaborn,
            # which currently uses this internal API.
            if kwargs["sharex"] is not self and kwargs["sharey"] is not self:
                raise ValueError("Twinned Axes may share only one axis")
        twin = self.figure.add_subplot(self.get_subplotspec(), *args, **kwargs)
        self.set_adjustable('datalim')
        twin.set_adjustable('datalim')
        self._twinned_axes.join(self, twin)
        return twin


# this here to support cartopy which was using a private part of the
# API to register their Axes subclasses.

# In 3.1 this should be changed to a dict subclass that warns on use
# In 3.3 to a dict subclass that raises a useful exception on use
# In 3.4 should be removed

# The slow timeline is to give cartopy enough time to get several
# release out before we break them.
_subplot_classes = {}


@functools.lru_cache(None)
def subplot_class_factory(axes_class=None):
    """
    Make a new class that inherits from `.SubplotBase` and the
    given axes_class (which is assumed to be a subclass of `.axes.Axes`).
    This is perhaps a little bit roundabout to make a new class on
    the fly like this, but it means that a new Subplot class does
    not have to be created for every type of Axes.
    """
    if axes_class is None:
        _api.warn_deprecated(
            "3.3", message="Support for passing None to subplot_class_factory "
            "is deprecated since %(since)s; explicitly pass the default Axes "
            "class instead. This will become an error %(removal)s.")
        axes_class = Axes
    try:
        # Avoid creating two different instances of GeoAxesSubplot...
        # Only a temporary backcompat fix.  This should be removed in
        # 3.4
        return next(cls for cls in SubplotBase.__subclasses__()
                    if cls.__bases__ == (SubplotBase, axes_class))
    except StopIteration:
        return type("%sSubplot" % axes_class.__name__,
                    (SubplotBase, axes_class),
                    {'_axes_class': axes_class})


Subplot = subplot_class_factory(Axes)  # Provided for backward compatibility.


def _picklable_subplot_class_constructor(axes_class):
    """
    Stub factory that returns an empty instance of the appropriate subplot
    class when called with an axes class. This is purely to allow pickling of
    Axes and Subplots.
    """
    subplot_class = subplot_class_factory(axes_class)
    return subplot_class.__new__(subplot_class)


docstring.interpd.update(Axes_kwdoc=martist.kwdoc(Axes))
docstring.dedent_interpd(Axes.__init__)

docstring.interpd.update(Subplot_kwdoc=martist.kwdoc(Axes))
