"""
Render to qt from agg.
"""

import ctypes

from matplotlib.transforms import Bbox

from .. import cbook
from .backend_agg import FigureCanvasAgg
from .backend_qt5 import (
    QtCore, QtGui, QtWidgets, _BackendQT5, FigureCanvasQT, FigureManagerQT,
    NavigationToolbar2QT, backend_version)
from .qt_compat import QT_API, _setDevicePixelRatio


class FigureCanvasQTAgg(FigureCanvasAgg, FigureCanvasQT):

    def __init__(self, figure=None):
        # Must pass 'figure' as kwarg to Qt base class.
        super().__init__(figure=figure)

    def paintEvent(self, event):
        """
        Copy the image from the Agg canvas to the qt.drawable.

        In Qt, all drawing should be done inside of here when a widget is
        shown onscreen.
        """
        self._draw_idle()  # Only does something if a draw is pending.

        # If the canvas does not have a renderer, then give up and wait for
        # FigureCanvasAgg.draw(self) to be called.
        if not hasattr(self, 'renderer'):
            return

        painter = QtGui.QPainter(self)
        try:
            # See documentation of QRect: bottom() and right() are off
            # by 1, so use left() + width() and top() + height().
            rect = event.rect()
            # scale rect dimensions using the screen dpi ratio to get
            # correct values for the Figure coordinates (rather than
            # QT5's coords)
            width = rect.width() * self._dpi_ratio
            height = rect.height() * self._dpi_ratio
            left, top = self.mouseEventCoords(rect.topLeft())
            # shift the "top" by the height of the image to get the
            # correct corner for our coordinate system
            bottom = top - height
            # same with the right side of the image
            right = left + width
            # create a buffer using the image bounding box
            bbox = Bbox([[left, bottom], [right, top]])
            reg = self.copy_from_bbox(bbox)
            buf = cbook._unmultiplied_rgba8888_to_premultiplied_argb32(
                memoryview(reg))

            # clear the widget canvas
            painter.eraseRect(rect)

            qimage = QtGui.QImage(buf, buf.shape[1], buf.shape[0],
                                  QtGui.QImage.Format_ARGB32_Premultiplied)
            _setDevicePixelRatio(qimage, self._dpi_ratio)
            # set origin using original QT coordinates
            origin = QtCore.QPoint(rect.left(), rect.top())
            painter.drawImage(origin, qimage)
            # Adjust the buf reference count to work around a memory
            # leak bug in QImage under PySide on Python 3.
            if QT_API in ('PySide', 'PySide2'):
                ctypes.c_long.from_address(id(buf)).value = 1

            self._draw_rect_callback(painter)
        finally:
            painter.end()

    def print_figure(self, *args, **kwargs):
        super().print_figure(*args, **kwargs)
        self.draw()


@_BackendQT5.export
class _BackendQT5Agg(_BackendQT5):
    FigureCanvas = FigureCanvasQTAgg
