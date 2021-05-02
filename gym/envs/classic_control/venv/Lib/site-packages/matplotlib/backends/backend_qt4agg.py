"""
Render to qt from agg
"""

from .. import _api
from .backend_qt5agg import (
    _BackendQT5Agg, FigureCanvasQTAgg, FigureManagerQT, NavigationToolbar2QT)


_api.warn_deprecated("3.3", name=__name__, obj_type="backend")


@_BackendQT5Agg.export
class _BackendQT4Agg(_BackendQT5Agg):
    class FigureCanvas(FigureCanvasQTAgg):
        required_interactive_framework = "qt4"
