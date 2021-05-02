from .. import _api
from .backend_qt5cairo import _BackendQT5Cairo, FigureCanvasQTCairo


_api.warn_deprecated("3.3", name=__name__, obj_type="backend")


@_BackendQT5Cairo.export
class _BackendQT4Cairo(_BackendQT5Cairo):
    class FigureCanvas(FigureCanvasQTCairo):
        required_interactive_framework = "qt4"
