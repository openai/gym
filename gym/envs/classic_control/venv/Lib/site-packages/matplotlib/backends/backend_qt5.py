import functools
import os
import signal
import sys
import traceback

import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2,
    TimerBase, cursors, ToolContainerBase, StatusbarBase, MouseButton)
import matplotlib.backends.qt_editor.figureoptions as figureoptions
from matplotlib.backends.qt_editor._formsubplottool import UiSubplotTool
from . import qt_compat
from .qt_compat import (
    QtCore, QtGui, QtWidgets, __version__, QT_API,
    _devicePixelRatioF, _isdeleted, _setDevicePixelRatio,
)

backend_version = __version__

# SPECIAL_KEYS are keys that do *not* return their unicode name
# instead they have manually specified names
SPECIAL_KEYS = {QtCore.Qt.Key_Control: 'control',
                QtCore.Qt.Key_Shift: 'shift',
                QtCore.Qt.Key_Alt: 'alt',
                QtCore.Qt.Key_Meta: 'meta',
                QtCore.Qt.Key_Super_L: 'super',
                QtCore.Qt.Key_Super_R: 'super',
                QtCore.Qt.Key_CapsLock: 'caps_lock',
                QtCore.Qt.Key_Return: 'enter',
                QtCore.Qt.Key_Left: 'left',
                QtCore.Qt.Key_Up: 'up',
                QtCore.Qt.Key_Right: 'right',
                QtCore.Qt.Key_Down: 'down',
                QtCore.Qt.Key_Escape: 'escape',
                QtCore.Qt.Key_F1: 'f1',
                QtCore.Qt.Key_F2: 'f2',
                QtCore.Qt.Key_F3: 'f3',
                QtCore.Qt.Key_F4: 'f4',
                QtCore.Qt.Key_F5: 'f5',
                QtCore.Qt.Key_F6: 'f6',
                QtCore.Qt.Key_F7: 'f7',
                QtCore.Qt.Key_F8: 'f8',
                QtCore.Qt.Key_F9: 'f9',
                QtCore.Qt.Key_F10: 'f10',
                QtCore.Qt.Key_F11: 'f11',
                QtCore.Qt.Key_F12: 'f12',
                QtCore.Qt.Key_Home: 'home',
                QtCore.Qt.Key_End: 'end',
                QtCore.Qt.Key_PageUp: 'pageup',
                QtCore.Qt.Key_PageDown: 'pagedown',
                QtCore.Qt.Key_Tab: 'tab',
                QtCore.Qt.Key_Backspace: 'backspace',
                QtCore.Qt.Key_Enter: 'enter',
                QtCore.Qt.Key_Insert: 'insert',
                QtCore.Qt.Key_Delete: 'delete',
                QtCore.Qt.Key_Pause: 'pause',
                QtCore.Qt.Key_SysReq: 'sysreq',
                QtCore.Qt.Key_Clear: 'clear', }
if sys.platform == 'darwin':
    # in OSX, the control and super (aka cmd/apple) keys are switched, so
    # switch them back.
    SPECIAL_KEYS.update({QtCore.Qt.Key_Control: 'cmd',  # cmd/apple key
                         QtCore.Qt.Key_Meta: 'control',
                         })
# Define which modifier keys are collected on keyboard events.
# Elements are (Modifier Flag, Qt Key) tuples.
# Order determines the modifier order (ctrl+alt+...) reported by Matplotlib.
_MODIFIER_KEYS = [
    (QtCore.Qt.ControlModifier, QtCore.Qt.Key_Control),
    (QtCore.Qt.AltModifier, QtCore.Qt.Key_Alt),
    (QtCore.Qt.ShiftModifier, QtCore.Qt.Key_Shift),
    (QtCore.Qt.MetaModifier, QtCore.Qt.Key_Meta),
]
cursord = {
    cursors.MOVE: QtCore.Qt.SizeAllCursor,
    cursors.HAND: QtCore.Qt.PointingHandCursor,
    cursors.POINTER: QtCore.Qt.ArrowCursor,
    cursors.SELECT_REGION: QtCore.Qt.CrossCursor,
    cursors.WAIT: QtCore.Qt.WaitCursor,
    }
SUPER = 0  # Deprecated.
ALT = 1  # Deprecated.
CTRL = 2  # Deprecated.
SHIFT = 3  # Deprecated.
MODIFIER_KEYS = [  # Deprecated.
    (SPECIAL_KEYS[key], mod, key) for mod, key in _MODIFIER_KEYS]


# make place holder
qApp = None


def _create_qApp():
    """
    Only one qApp can exist at a time, so check before creating one.
    """
    global qApp

    if qApp is None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            # display_is_valid returns False only if on Linux and neither X11
            # nor Wayland display can be opened.
            if not mpl._c_internal_utils.display_is_valid():
                raise RuntimeError('Invalid DISPLAY variable')
            try:
                QtWidgets.QApplication.setAttribute(
                    QtCore.Qt.AA_EnableHighDpiScaling)
            except AttributeError:  # Attribute only exists for Qt>=5.6.
                pass
            try:
                QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
                    QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
            except AttributeError:  # Added in Qt>=5.14.
                pass
            qApp = QtWidgets.QApplication(["matplotlib"])
            qApp.lastWindowClosed.connect(qApp.quit)
            cbook._setup_new_guiapp()
        else:
            qApp = app

    try:
        qApp.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    except AttributeError:
        pass


def _allow_super_init(__init__):
    """
    Decorator for ``__init__`` to allow ``super().__init__`` on PyQt4/PySide2.
    """

    if QT_API == "PyQt5":

        return __init__

    else:
        # To work around lack of cooperative inheritance in PyQt4, PySide,
        # and PySide2, when calling FigureCanvasQT.__init__, we temporarily
        # patch QWidget.__init__ by a cooperative version, that first calls
        # QWidget.__init__ with no additional arguments, and then finds the
        # next class in the MRO with an __init__ that does support cooperative
        # inheritance (i.e., not defined by the PyQt4, PySide, PySide2, sip
        # or Shiboken packages), and manually call its `__init__`, once again
        # passing the additional arguments.

        qwidget_init = QtWidgets.QWidget.__init__

        def cooperative_qwidget_init(self, *args, **kwargs):
            qwidget_init(self)
            mro = type(self).__mro__
            next_coop_init = next(
                cls for cls in mro[mro.index(QtWidgets.QWidget) + 1:]
                if cls.__module__.split(".")[0] not in [
                    "PyQt4", "sip", "PySide", "PySide2", "Shiboken"])
            next_coop_init.__init__(self, *args, **kwargs)

        @functools.wraps(__init__)
        def wrapper(self, *args, **kwargs):
            with cbook._setattr_cm(QtWidgets.QWidget,
                                   __init__=cooperative_qwidget_init):
                __init__(self, *args, **kwargs)

        return wrapper


class TimerQT(TimerBase):
    """Subclass of `.TimerBase` using QTimer events."""

    def __init__(self, *args, **kwargs):
        # Create a new timer and connect the timeout() signal to the
        # _on_timer method.
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._on_timer)
        super().__init__(*args, **kwargs)

    def __del__(self):
        # The check for deletedness is needed to avoid an error at animation
        # shutdown with PySide2.
        if not _isdeleted(self._timer):
            self._timer_stop()

    def _timer_set_single_shot(self):
        self._timer.setSingleShot(self._single)

    def _timer_set_interval(self):
        self._timer.setInterval(self._interval)

    def _timer_start(self):
        self._timer.start()

    def _timer_stop(self):
        self._timer.stop()


class FigureCanvasQT(QtWidgets.QWidget, FigureCanvasBase):
    required_interactive_framework = "qt5"
    _timer_cls = TimerQT

    # map Qt button codes to MouseEvent's ones:
    buttond = {QtCore.Qt.LeftButton: MouseButton.LEFT,
               QtCore.Qt.MidButton: MouseButton.MIDDLE,
               QtCore.Qt.RightButton: MouseButton.RIGHT,
               QtCore.Qt.XButton1: MouseButton.BACK,
               QtCore.Qt.XButton2: MouseButton.FORWARD,
               }

    @_allow_super_init
    def __init__(self, figure=None):
        _create_qApp()
        super().__init__(figure=figure)

        # We don't want to scale up the figure DPI more than once.
        # Note, we don't handle a signal for changing DPI yet.
        self.figure._original_dpi = self.figure.dpi
        self._update_figure_dpi()
        # In cases with mixed resolution displays, we need to be careful if the
        # dpi_ratio changes - in this case we need to resize the canvas
        # accordingly.
        self._dpi_ratio_prev = self._dpi_ratio

        self._draw_pending = False
        self._is_drawing = False
        self._draw_rect_callback = lambda painter: None

        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        self.setMouseTracking(True)
        self.resize(*self.get_width_height())

        palette = QtGui.QPalette(QtCore.Qt.white)
        self.setPalette(palette)

    def _update_figure_dpi(self):
        dpi = self._dpi_ratio * self.figure._original_dpi
        self.figure._set_dpi(dpi, forward=False)

    @property
    def _dpi_ratio(self):
        return _devicePixelRatioF(self)

    def _update_pixel_ratio(self):
        # We need to be careful in cases with mixed resolution displays if
        # dpi_ratio changes.
        if self._dpi_ratio != self._dpi_ratio_prev:
            # We need to update the figure DPI.
            self._update_figure_dpi()
            self._dpi_ratio_prev = self._dpi_ratio
            # The easiest way to resize the canvas is to emit a resizeEvent
            # since we implement all the logic for resizing the canvas for
            # that event.
            event = QtGui.QResizeEvent(self.size(), self.size())
            self.resizeEvent(event)
            # resizeEvent triggers a paintEvent itself, so we exit this one
            # (after making sure that the event is immediately handled).

    def _update_screen(self, screen):
        # Handler for changes to a window's attached screen.
        self._update_pixel_ratio()
        if screen is not None:
            screen.physicalDotsPerInchChanged.connect(self._update_pixel_ratio)
            screen.logicalDotsPerInchChanged.connect(self._update_pixel_ratio)

    def showEvent(self, event):
        # Set up correct pixel ratio, and connect to any signal changes for it,
        # once the window is shown (and thus has these attributes).
        window = self.window().windowHandle()
        window.screenChanged.connect(self._update_screen)
        self._update_screen(window.screen())

    def get_width_height(self):
        w, h = FigureCanvasBase.get_width_height(self)
        return int(w / self._dpi_ratio), int(h / self._dpi_ratio)

    def enterEvent(self, event):
        try:
            x, y = self.mouseEventCoords(event.pos())
        except AttributeError:
            # the event from PyQt4 does not include the position
            x = y = None
        FigureCanvasBase.enter_notify_event(self, guiEvent=event, xy=(x, y))

    def leaveEvent(self, event):
        QtWidgets.QApplication.restoreOverrideCursor()
        FigureCanvasBase.leave_notify_event(self, guiEvent=event)

    def mouseEventCoords(self, pos):
        """
        Calculate mouse coordinates in physical pixels.

        Qt5 use logical pixels, but the figure is scaled to physical
        pixels for rendering.  Transform to physical pixels so that
        all of the down-stream transforms work as expected.

        Also, the origin is different and needs to be corrected.
        """
        dpi_ratio = self._dpi_ratio
        x = pos.x()
        # flip y so y=0 is bottom of canvas
        y = self.figure.bbox.height / dpi_ratio - pos.y()
        return x * dpi_ratio, y * dpi_ratio

    def mousePressEvent(self, event):
        x, y = self.mouseEventCoords(event.pos())
        button = self.buttond.get(event.button())
        if button is not None:
            FigureCanvasBase.button_press_event(self, x, y, button,
                                                guiEvent=event)

    def mouseDoubleClickEvent(self, event):
        x, y = self.mouseEventCoords(event.pos())
        button = self.buttond.get(event.button())
        if button is not None:
            FigureCanvasBase.button_press_event(self, x, y,
                                                button, dblclick=True,
                                                guiEvent=event)

    def mouseMoveEvent(self, event):
        x, y = self.mouseEventCoords(event)
        FigureCanvasBase.motion_notify_event(self, x, y, guiEvent=event)

    def mouseReleaseEvent(self, event):
        x, y = self.mouseEventCoords(event)
        button = self.buttond.get(event.button())
        if button is not None:
            FigureCanvasBase.button_release_event(self, x, y, button,
                                                  guiEvent=event)

    if QtCore.qVersion() >= "5.":
        def wheelEvent(self, event):
            x, y = self.mouseEventCoords(event)
            # from QWheelEvent::delta doc
            if event.pixelDelta().x() == 0 and event.pixelDelta().y() == 0:
                steps = event.angleDelta().y() / 120
            else:
                steps = event.pixelDelta().y()
            if steps:
                FigureCanvasBase.scroll_event(
                    self, x, y, steps, guiEvent=event)
    else:
        def wheelEvent(self, event):
            x = event.x()
            # flipy so y=0 is bottom of canvas
            y = self.figure.bbox.height - event.y()
            # from QWheelEvent::delta doc
            steps = event.delta() / 120
            if event.orientation() == QtCore.Qt.Vertical:
                FigureCanvasBase.scroll_event(
                    self, x, y, steps, guiEvent=event)

    def keyPressEvent(self, event):
        key = self._get_key(event)
        if key is not None:
            FigureCanvasBase.key_press_event(self, key, guiEvent=event)

    def keyReleaseEvent(self, event):
        key = self._get_key(event)
        if key is not None:
            FigureCanvasBase.key_release_event(self, key, guiEvent=event)

    def resizeEvent(self, event):
        w = event.size().width() * self._dpi_ratio
        h = event.size().height() * self._dpi_ratio
        dpival = self.figure.dpi
        winch = w / dpival
        hinch = h / dpival
        self.figure.set_size_inches(winch, hinch, forward=False)
        # pass back into Qt to let it finish
        QtWidgets.QWidget.resizeEvent(self, event)
        # emit our resize events
        FigureCanvasBase.resize_event(self)

    def sizeHint(self):
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)

    def minumumSizeHint(self):
        return QtCore.QSize(10, 10)

    def _get_key(self, event):
        event_key = event.key()
        event_mods = int(event.modifiers())  # actually a bitmask

        # get names of the pressed modifier keys
        # 'control' is named 'control' when a standalone key, but 'ctrl' when a
        # modifier
        # bit twiddling to pick out modifier keys from event_mods bitmask,
        # if event_key is a MODIFIER, it should not be duplicated in mods
        mods = [SPECIAL_KEYS[key].replace('control', 'ctrl')
                for mod, key in _MODIFIER_KEYS
                if event_key != key and event_mods & mod]
        try:
            # for certain keys (enter, left, backspace, etc) use a word for the
            # key, rather than unicode
            key = SPECIAL_KEYS[event_key]
        except KeyError:
            # unicode defines code points up to 0x10ffff (sys.maxunicode)
            # QT will use Key_Codes larger than that for keyboard keys that are
            # are not unicode characters (like multimedia keys)
            # skip these
            # if you really want them, you should add them to SPECIAL_KEYS
            if event_key > sys.maxunicode:
                return None

            key = chr(event_key)
            # qt delivers capitalized letters.  fix capitalization
            # note that capslock is ignored
            if 'shift' in mods:
                mods.remove('shift')
            else:
                key = key.lower()

        return '+'.join(mods + [key])

    def flush_events(self):
        # docstring inherited
        qApp.processEvents()

    def start_event_loop(self, timeout=0):
        # docstring inherited
        if hasattr(self, "_event_loop") and self._event_loop.isRunning():
            raise RuntimeError("Event loop already running")
        self._event_loop = event_loop = QtCore.QEventLoop()
        if timeout > 0:
            timer = QtCore.QTimer.singleShot(int(timeout * 1000),
                                             event_loop.quit)
        event_loop.exec_()

    def stop_event_loop(self, event=None):
        # docstring inherited
        if hasattr(self, "_event_loop"):
            self._event_loop.quit()

    def draw(self):
        """Render the figure, and queue a request for a Qt draw."""
        # The renderer draw is done here; delaying causes problems with code
        # that uses the result of the draw() to update plot elements.
        if self._is_drawing:
            return
        with cbook._setattr_cm(self, _is_drawing=True):
            super().draw()
        self.update()

    def draw_idle(self):
        """Queue redraw of the Agg buffer and request Qt paintEvent."""
        # The Agg draw needs to be handled by the same thread Matplotlib
        # modifies the scene graph from. Post Agg draw request to the
        # current event loop in order to ensure thread affinity and to
        # accumulate multiple draw requests from event handling.
        # TODO: queued signal connection might be safer than singleShot
        if not (getattr(self, '_draw_pending', False) or
                getattr(self, '_is_drawing', False)):
            self._draw_pending = True
            QtCore.QTimer.singleShot(0, self._draw_idle)

    def blit(self, bbox=None):
        # docstring inherited
        if bbox is None and self.figure:
            bbox = self.figure.bbox  # Blit the entire canvas if bbox is None.
        # repaint uses logical pixels, not physical pixels like the renderer.
        l, b, w, h = [int(pt / self._dpi_ratio) for pt in bbox.bounds]
        t = b + h
        self.repaint(l, self.rect().height() - t, w, h)

    def _draw_idle(self):
        with self._idle_draw_cntx():
            if not self._draw_pending:
                return
            self._draw_pending = False
            if self.height() < 0 or self.width() < 0:
                return
            try:
                self.draw()
            except Exception:
                # Uncaught exceptions are fatal for PyQt5, so catch them.
                traceback.print_exc()

    def drawRectangle(self, rect):
        # Draw the zoom rectangle to the QPainter.  _draw_rect_callback needs
        # to be called at the end of paintEvent.
        if rect is not None:
            x0, y0, w, h = [int(pt / self._dpi_ratio) for pt in rect]
            x1 = x0 + w
            y1 = y0 + h
            def _draw_rect_callback(painter):
                pen = QtGui.QPen(QtCore.Qt.black, 1 / self._dpi_ratio)
                pen.setDashPattern([3, 3])
                for color, offset in [
                        (QtCore.Qt.black, 0), (QtCore.Qt.white, 3)]:
                    pen.setDashOffset(offset)
                    pen.setColor(color)
                    painter.setPen(pen)
                    # Draw the lines from x0, y0 towards x1, y1 so that the
                    # dashes don't "jump" when moving the zoom box.
                    painter.drawLine(x0, y0, x0, y1)
                    painter.drawLine(x0, y0, x1, y0)
                    painter.drawLine(x0, y1, x1, y1)
                    painter.drawLine(x1, y0, x1, y1)
        else:
            def _draw_rect_callback(painter):
                return
        self._draw_rect_callback = _draw_rect_callback
        self.update()


class MainWindow(QtWidgets.QMainWindow):
    closing = QtCore.Signal()

    def closeEvent(self, event):
        self.closing.emit()
        super().closeEvent(event)


class FigureManagerQT(FigureManagerBase):
    """
    Attributes
    ----------
    canvas : `FigureCanvas`
        The FigureCanvas instance
    num : int or str
        The Figure number
    toolbar : qt.QToolBar
        The qt.QToolBar
    window : qt.QMainWindow
        The qt.QMainWindow
    """

    def __init__(self, canvas, num):
        self.window = MainWindow()
        super().__init__(canvas, num)
        self.window.closing.connect(canvas.close_event)
        self.window.closing.connect(self._widgetclosed)

        image = str(cbook._get_data_path('images/matplotlib.svg'))
        self.window.setWindowIcon(QtGui.QIcon(image))

        self.window._destroying = False

        self.toolbar = self._get_toolbar(self.canvas, self.window)

        if self.toolmanager:
            backend_tools.add_tools_to_manager(self.toolmanager)
            if self.toolbar:
                backend_tools.add_tools_to_container(self.toolbar)

        if self.toolbar:
            self.window.addToolBar(self.toolbar)
            tbs_height = self.toolbar.sizeHint().height()
        else:
            tbs_height = 0

        # resize the main window so it will display the canvas with the
        # requested size:
        cs = canvas.sizeHint()
        cs_height = cs.height()
        height = cs_height + tbs_height
        self.window.resize(cs.width(), height)

        self.window.setCentralWidget(self.canvas)

        if mpl.is_interactive():
            self.window.show()
            self.canvas.draw_idle()

        # Give the keyboard focus to the figure instead of the manager:
        # StrongFocus accepts both tab and click to focus and will enable the
        # canvas to process event without clicking.
        # https://doc.qt.io/qt-5/qt.html#FocusPolicy-enum
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.canvas.setFocus()

        self.window.raise_()

    def full_screen_toggle(self):
        if self.window.isFullScreen():
            self.window.showNormal()
        else:
            self.window.showFullScreen()

    def _widgetclosed(self):
        if self.window._destroying:
            return
        self.window._destroying = True
        try:
            Gcf.destroy(self)
        except AttributeError:
            pass
            # It seems that when the python session is killed,
            # Gcf can get destroyed before the Gcf.destroy
            # line is run, leading to a useless AttributeError.

    def _get_toolbar(self, canvas, parent):
        # must be inited after the window, drawingArea and figure
        # attrs are set
        if mpl.rcParams['toolbar'] == 'toolbar2':
            toolbar = NavigationToolbar2QT(canvas, parent, True)
        elif mpl.rcParams['toolbar'] == 'toolmanager':
            toolbar = ToolbarQt(self.toolmanager, self.window)
        else:
            toolbar = None
        return toolbar

    def resize(self, width, height):
        # these are Qt methods so they return sizes in 'virtual' pixels
        # so we do not need to worry about dpi scaling here.
        extra_width = self.window.width() - self.canvas.width()
        extra_height = self.window.height() - self.canvas.height()
        self.canvas.resize(width, height)
        self.window.resize(width + extra_width, height + extra_height)

    def show(self):
        self.window.show()
        if mpl.rcParams['figure.raise_window']:
            self.window.activateWindow()
            self.window.raise_()

    def destroy(self, *args):
        # check for qApp first, as PySide deletes it in its atexit handler
        if QtWidgets.QApplication.instance() is None:
            return
        if self.window._destroying:
            return
        self.window._destroying = True
        if self.toolbar:
            self.toolbar.destroy()
        self.window.close()

    def get_window_title(self):
        return self.window.windowTitle()

    def set_window_title(self, title):
        self.window.setWindowTitle(title)


class NavigationToolbar2QT(NavigationToolbar2, QtWidgets.QToolBar):
    message = QtCore.Signal(str)

    toolitems = [*NavigationToolbar2.toolitems]
    toolitems.insert(
        # Add 'customize' action after 'subplots'
        [name for name, *_ in toolitems].index("Subplots") + 1,
        ("Customize", "Edit axis, curve and image parameters",
         "qt4_editor_options", "edit_parameters"))

    def __init__(self, canvas, parent, coordinates=True):
        """coordinates: should we show the coordinates on the right?"""
        QtWidgets.QToolBar.__init__(self, parent)
        self.setAllowedAreas(
            QtCore.Qt.TopToolBarArea | QtCore.Qt.BottomToolBarArea)

        self.coordinates = coordinates
        self._actions = {}  # mapping of toolitem method names to QActions.

        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.addSeparator()
            else:
                a = self.addAction(self._icon(image_file + '.png'),
                                   text, getattr(self, callback))
                self._actions[callback] = a
                if callback in ['zoom', 'pan']:
                    a.setCheckable(True)
                if tooltip_text is not None:
                    a.setToolTip(tooltip_text)

        # Add the (x, y) location widget at the right side of the toolbar
        # The stretch factor is 1 which means any resizing of the toolbar
        # will resize this label instead of the buttons.
        if self.coordinates:
            self.locLabel = QtWidgets.QLabel("", self)
            self.locLabel.setAlignment(
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.locLabel.setSizePolicy(
                QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                      QtWidgets.QSizePolicy.Ignored))
            labelAction = self.addWidget(self.locLabel)
            labelAction.setVisible(True)

        NavigationToolbar2.__init__(self, canvas)

    @_api.deprecated("3.3", alternative="self.canvas.parent()")
    @property
    def parent(self):
        return self.canvas.parent()

    @_api.deprecated("3.3", alternative="self.canvas.setParent()")
    @parent.setter
    def parent(self, value):
        pass

    @_api.deprecated(
        "3.3", alternative="os.path.join(mpl.get_data_path(), 'images')")
    @property
    def basedir(self):
        return str(cbook._get_data_path('images'))

    def _icon(self, name):
        """
        Construct a `.QIcon` from an image file *name*, including the extension
        and relative to Matplotlib's "images" data directory.
        """
        if QtCore.qVersion() >= '5.':
            name = name.replace('.png', '_large.png')
        pm = QtGui.QPixmap(str(cbook._get_data_path('images', name)))
        _setDevicePixelRatio(pm, _devicePixelRatioF(self))
        if self.palette().color(self.backgroundRole()).value() < 128:
            icon_color = self.palette().color(self.foregroundRole())
            mask = pm.createMaskFromColor(QtGui.QColor('black'),
                                          QtCore.Qt.MaskOutColor)
            pm.fill(icon_color)
            pm.setMask(mask)
        return QtGui.QIcon(pm)

    def edit_parameters(self):
        axes = self.canvas.figure.get_axes()
        if not axes:
            QtWidgets.QMessageBox.warning(
                self.canvas.parent(), "Error", "There are no axes to edit.")
            return
        elif len(axes) == 1:
            ax, = axes
        else:
            titles = [
                ax.get_label() or
                ax.get_title() or
                " - ".join(filter(None, [ax.get_xlabel(), ax.get_ylabel()])) or
                f"<anonymous {type(ax).__name__}>"
                for ax in axes]
            duplicate_titles = [
                title for title in titles if titles.count(title) > 1]
            for i, ax in enumerate(axes):
                if titles[i] in duplicate_titles:
                    titles[i] += f" (id: {id(ax):#x})"  # Deduplicate titles.
            item, ok = QtWidgets.QInputDialog.getItem(
                self.canvas.parent(),
                'Customize', 'Select axes:', titles, 0, False)
            if not ok:
                return
            ax = axes[titles.index(item)]
        figureoptions.figure_edit(ax, self)

    def _update_buttons_checked(self):
        # sync button checkstates to match active mode
        if 'pan' in self._actions:
            self._actions['pan'].setChecked(self.mode.name == 'PAN')
        if 'zoom' in self._actions:
            self._actions['zoom'].setChecked(self.mode.name == 'ZOOM')

    def pan(self, *args):
        super().pan(*args)
        self._update_buttons_checked()

    def zoom(self, *args):
        super().zoom(*args)
        self._update_buttons_checked()

    def set_message(self, s):
        self.message.emit(s)
        if self.coordinates:
            self.locLabel.setText(s)

    def set_cursor(self, cursor):
        self.canvas.setCursor(cursord[cursor])

    def draw_rubberband(self, event, x0, y0, x1, y1):
        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0
        rect = [int(val) for val in (x0, y0, x1 - x0, y1 - y0)]
        self.canvas.drawRectangle(rect)

    def remove_rubberband(self):
        self.canvas.drawRectangle(None)

    def configure_subplots(self):
        image = str(cbook._get_data_path('images/matplotlib.png'))
        dia = SubplotToolQt(self.canvas.figure, self.canvas.parent())
        dia.setWindowIcon(QtGui.QIcon(image))
        dia.exec_()

    def save_figure(self, *args):
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        default_filetype = self.canvas.get_default_filetype()

        startpath = os.path.expanduser(mpl.rcParams['savefig.directory'])
        start = os.path.join(startpath, self.canvas.get_default_filename())
        filters = []
        selectedFilter = None
        for name, exts in sorted_filetypes:
            exts_list = " ".join(['*.%s' % ext for ext in exts])
            filter = '%s (%s)' % (name, exts_list)
            if default_filetype in exts:
                selectedFilter = filter
            filters.append(filter)
        filters = ';;'.join(filters)

        fname, filter = qt_compat._getSaveFileName(
            self.canvas.parent(), "Choose a filename to save to", start,
            filters, selectedFilter)
        if fname:
            # Save dir for next time, unless empty str (i.e., use cwd).
            if startpath != "":
                mpl.rcParams['savefig.directory'] = os.path.dirname(fname)
            try:
                self.canvas.figure.savefig(fname)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error saving file", str(e),
                    QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.NoButton)

    def set_history_buttons(self):
        can_backward = self._nav_stack._pos > 0
        can_forward = self._nav_stack._pos < len(self._nav_stack._elements) - 1
        if 'back' in self._actions:
            self._actions['back'].setEnabled(can_backward)
        if 'forward' in self._actions:
            self._actions['forward'].setEnabled(can_forward)


class SubplotToolQt(UiSubplotTool):
    def __init__(self, targetfig, parent):
        super().__init__(None)

        self._figure = targetfig

        for lower, higher in [("bottom", "top"), ("left", "right")]:
            self._widgets[lower].valueChanged.connect(
                lambda val: self._widgets[higher].setMinimum(val + .001))
            self._widgets[higher].valueChanged.connect(
                lambda val: self._widgets[lower].setMaximum(val - .001))

        self._attrs = ["top", "bottom", "left", "right", "hspace", "wspace"]
        self._defaults = {attr: vars(self._figure.subplotpars)[attr]
                          for attr in self._attrs}

        # Set values after setting the range callbacks, but before setting up
        # the redraw callbacks.
        self._reset()

        for attr in self._attrs:
            self._widgets[attr].valueChanged.connect(self._on_value_changed)
        for action, method in [("Export values", self._export_values),
                               ("Tight layout", self._tight_layout),
                               ("Reset", self._reset),
                               ("Close", self.close)]:
            self._widgets[action].clicked.connect(method)

    def _export_values(self):
        # Explicitly round to 3 decimals (which is also the spinbox precision)
        # to avoid numbers of the form 0.100...001.
        dialog = QtWidgets.QDialog()
        layout = QtWidgets.QVBoxLayout()
        dialog.setLayout(layout)
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        layout.addWidget(text)
        text.setPlainText(
            ",\n".join("{}={:.3}".format(attr, self._widgets[attr].value())
                       for attr in self._attrs))
        # Adjust the height of the text widget to fit the whole text, plus
        # some padding.
        size = text.maximumSize()
        size.setHeight(
            QtGui.QFontMetrics(text.document().defaultFont())
            .size(0, text.toPlainText()).height() + 20)
        text.setMaximumSize(size)
        dialog.exec_()

    def _on_value_changed(self):
        self._figure.subplots_adjust(**{attr: self._widgets[attr].value()
                                        for attr in self._attrs})
        self._figure.canvas.draw_idle()

    def _tight_layout(self):
        self._figure.tight_layout()
        for attr in self._attrs:
            widget = self._widgets[attr]
            widget.blockSignals(True)
            widget.setValue(vars(self._figure.subplotpars)[attr])
            widget.blockSignals(False)
        self._figure.canvas.draw_idle()

    def _reset(self):
        for attr, value in self._defaults.items():
            self._widgets[attr].setValue(value)


class ToolbarQt(ToolContainerBase, QtWidgets.QToolBar):
    def __init__(self, toolmanager, parent):
        ToolContainerBase.__init__(self, toolmanager)
        QtWidgets.QToolBar.__init__(self, parent)
        self.setAllowedAreas(
            QtCore.Qt.TopToolBarArea | QtCore.Qt.BottomToolBarArea)
        message_label = QtWidgets.QLabel("")
        message_label.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        message_label.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Ignored))
        self._message_action = self.addWidget(message_label)
        self._toolitems = {}
        self._groups = {}

    def add_toolitem(
            self, name, group, position, image_file, description, toggle):

        button = QtWidgets.QToolButton(self)
        if image_file:
            button.setIcon(NavigationToolbar2QT._icon(self, image_file))
        button.setText(name)
        if description:
            button.setToolTip(description)

        def handler():
            self.trigger_tool(name)
        if toggle:
            button.setCheckable(True)
            button.toggled.connect(handler)
        else:
            button.clicked.connect(handler)

        self._toolitems.setdefault(name, [])
        self._add_to_group(group, name, button, position)
        self._toolitems[name].append((button, handler))

    def _add_to_group(self, group, name, button, position):
        gr = self._groups.get(group, [])
        if not gr:
            sep = self.insertSeparator(self._message_action)
            gr.append(sep)
        before = gr[position]
        widget = self.insertWidget(before, button)
        gr.insert(position, widget)
        self._groups[group] = gr

    def toggle_toolitem(self, name, toggled):
        if name not in self._toolitems:
            return
        for button, handler in self._toolitems[name]:
            button.toggled.disconnect(handler)
            button.setChecked(toggled)
            button.toggled.connect(handler)

    def remove_toolitem(self, name):
        for button, handler in self._toolitems[name]:
            button.setParent(None)
        del self._toolitems[name]

    def set_message(self, s):
        self.widgetForAction(self._message_action).setText(s)


@_api.deprecated("3.3")
class StatusbarQt(StatusbarBase, QtWidgets.QLabel):
    def __init__(self, window, *args, **kwargs):
        StatusbarBase.__init__(self, *args, **kwargs)
        QtWidgets.QLabel.__init__(self)
        window.statusBar().addWidget(self)

    def set_message(self, s):
        self.setText(s)


class ConfigureSubplotsQt(backend_tools.ConfigureSubplotsBase):
    def trigger(self, *args):
        NavigationToolbar2QT.configure_subplots(
            self._make_classic_style_pseudo_toolbar())


class SaveFigureQt(backend_tools.SaveFigureBase):
    def trigger(self, *args):
        NavigationToolbar2QT.save_figure(
            self._make_classic_style_pseudo_toolbar())


class SetCursorQt(backend_tools.SetCursorBase):
    def set_cursor(self, cursor):
        NavigationToolbar2QT.set_cursor(
            self._make_classic_style_pseudo_toolbar(), cursor)


class RubberbandQt(backend_tools.RubberbandBase):
    def draw_rubberband(self, x0, y0, x1, y1):
        NavigationToolbar2QT.draw_rubberband(
            self._make_classic_style_pseudo_toolbar(), None, x0, y0, x1, y1)

    def remove_rubberband(self):
        NavigationToolbar2QT.remove_rubberband(
            self._make_classic_style_pseudo_toolbar())


class HelpQt(backend_tools.ToolHelpBase):
    def trigger(self, *args):
        QtWidgets.QMessageBox.information(None, "Help", self._get_help_html())


class ToolCopyToClipboardQT(backend_tools.ToolCopyToClipboardBase):
    def trigger(self, *args, **kwargs):
        pixmap = self.canvas.grab()
        qApp.clipboard().setPixmap(pixmap)


backend_tools.ToolSaveFigure = SaveFigureQt
backend_tools.ToolConfigureSubplots = ConfigureSubplotsQt
backend_tools.ToolSetCursor = SetCursorQt
backend_tools.ToolRubberband = RubberbandQt
backend_tools.ToolHelp = HelpQt
backend_tools.ToolCopyToClipboard = ToolCopyToClipboardQT


@_Backend.export
class _BackendQT5(_Backend):
    FigureCanvas = FigureCanvasQT
    FigureManager = FigureManagerQT

    @staticmethod
    def mainloop():
        old_signal = signal.getsignal(signal.SIGINT)
        # allow SIGINT exceptions to close the plot window.
        is_python_signal_handler = old_signal is not None
        if is_python_signal_handler:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        try:
            qApp.exec_()
        finally:
            # reset the SIGINT exception handler
            if is_python_signal_handler:
                signal.signal(signal.SIGINT, old_signal)
