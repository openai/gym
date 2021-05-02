import mpl_toolkits.axes_grid1.axes_grid as axes_grid_orig
from .axislines import Axes


class CbarAxes(axes_grid_orig.CbarAxesBase, Axes):
    pass


class Grid(axes_grid_orig.Grid):
    _defaultAxesClass = Axes


class ImageGrid(axes_grid_orig.ImageGrid):
    _defaultAxesClass = Axes
    _defaultCbarAxesClass = CbarAxes


AxesGrid = ImageGrid
