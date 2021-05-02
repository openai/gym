from .. import _api
from .backend_qt5 import (
    backend_version, SPECIAL_KEYS,
    SUPER, ALT, CTRL, SHIFT, MODIFIER_KEYS,  # These are deprecated.
    cursord, _create_qApp, _BackendQT5, TimerQT, MainWindow, FigureCanvasQT,
    FigureManagerQT, NavigationToolbar2QT, SubplotToolQt, exception_handler)


_api.warn_deprecated("3.3", name=__name__, obj_type="backend")


@_BackendQT5.export
class _BackendQT4(_BackendQT5):
    class FigureCanvas(FigureCanvasQT):
        required_interactive_framework = "qt4"
