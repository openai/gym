"""
Qt binding and backend selector.

The selection logic is as follows:
- if any of PyQt5, PySide2, PyQt4 or PySide have already been imported
  (checked in that order), use it;
- otherwise, if the QT_API environment variable (used by Enthought) is set, use
  it to determine which binding to use (but do not change the backend based on
  it; i.e. if the Qt5Agg backend is requested but QT_API is set to "pyqt4",
  then actually use Qt5 with PyQt5 or PySide2 (whichever can be imported);
- otherwise, use whatever the rcParams indicate.

Support for PyQt4 is deprecated.
"""

from distutils.version import LooseVersion
import os
import platform
import sys

import matplotlib as mpl
from matplotlib import _api


QT_API_PYQT5 = "PyQt5"
QT_API_PYSIDE2 = "PySide2"
QT_API_PYQTv2 = "PyQt4v2"
QT_API_PYSIDE = "PySide"
QT_API_PYQT = "PyQt4"   # Use the old sip v1 API (Py3 defaults to v2).
QT_API_ENV = os.environ.get("QT_API")
if QT_API_ENV is not None:
    QT_API_ENV = QT_API_ENV.lower()
# Mapping of QT_API_ENV to requested binding.  ETS does not support PyQt4v1.
# (https://github.com/enthought/pyface/blob/master/pyface/qt/__init__.py)
_ETS = {"pyqt5": QT_API_PYQT5, "pyside2": QT_API_PYSIDE2,
        "pyqt": QT_API_PYQTv2, "pyside": QT_API_PYSIDE,
        None: None}
# First, check if anything is already imported.
if "PyQt5.QtCore" in sys.modules:
    QT_API = QT_API_PYQT5
elif "PySide2.QtCore" in sys.modules:
    QT_API = QT_API_PYSIDE2
elif "PyQt4.QtCore" in sys.modules:
    QT_API = QT_API_PYQTv2
elif "PySide.QtCore" in sys.modules:
    QT_API = QT_API_PYSIDE
# Otherwise, check the QT_API environment variable (from Enthought).  This can
# only override the binding, not the backend (in other words, we check that the
# requested backend actually matches).  Use dict.__getitem__ to avoid
# triggering backend resolution (which can result in a partially but
# incompletely imported backend_qt5).
elif dict.__getitem__(mpl.rcParams, "backend") in ["Qt5Agg", "Qt5Cairo"]:
    if QT_API_ENV in ["pyqt5", "pyside2"]:
        QT_API = _ETS[QT_API_ENV]
    else:
        QT_API = None
elif dict.__getitem__(mpl.rcParams, "backend") in ["Qt4Agg", "Qt4Cairo"]:
    if QT_API_ENV in ["pyqt4", "pyside"]:
        QT_API = _ETS[QT_API_ENV]
    else:
        QT_API = None
# A non-Qt backend was selected but we still got there (possible, e.g., when
# fully manually embedding Matplotlib in a Qt app without using pyplot).
else:
    try:
        QT_API = _ETS[QT_API_ENV]
    except KeyError as err:
        raise RuntimeError(
            "The environment variable QT_API has the unrecognized value {!r};"
            "valid values are 'pyqt5', 'pyside2', 'pyqt', and "
            "'pyside'") from err


def _setup_pyqt5():
    global QtCore, QtGui, QtWidgets, __version__, is_pyqt5, \
        _isdeleted, _getSaveFileName

    if QT_API == QT_API_PYQT5:
        from PyQt5 import QtCore, QtGui, QtWidgets
        import sip
        __version__ = QtCore.PYQT_VERSION_STR
        QtCore.Signal = QtCore.pyqtSignal
        QtCore.Slot = QtCore.pyqtSlot
        QtCore.Property = QtCore.pyqtProperty
        _isdeleted = sip.isdeleted
    elif QT_API == QT_API_PYSIDE2:
        from PySide2 import QtCore, QtGui, QtWidgets, __version__
        import shiboken2
        def _isdeleted(obj): return not shiboken2.isValid(obj)
    else:
        raise ValueError("Unexpected value for the 'backend.qt5' rcparam")
    _getSaveFileName = QtWidgets.QFileDialog.getSaveFileName

    @_api.deprecated("3.3", alternative="QtCore.qVersion()")
    def is_pyqt5():
        return True


def _setup_pyqt4():
    global QtCore, QtGui, QtWidgets, __version__, is_pyqt5, \
        _isdeleted, _getSaveFileName

    def _setup_pyqt4_internal(api):
        global QtCore, QtGui, QtWidgets, \
            __version__, is_pyqt5, _isdeleted, _getSaveFileName
        # List of incompatible APIs:
        # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
        _sip_apis = ["QDate", "QDateTime", "QString", "QTextStream", "QTime",
                     "QUrl", "QVariant"]
        try:
            import sip
        except ImportError:
            pass
        else:
            for _sip_api in _sip_apis:
                try:
                    sip.setapi(_sip_api, api)
                except ValueError:
                    pass
        from PyQt4 import QtCore, QtGui
        import sip  # Always succeeds *after* importing PyQt4.
        __version__ = QtCore.PYQT_VERSION_STR
        # PyQt 4.6 introduced getSaveFileNameAndFilter:
        # https://riverbankcomputing.com/news/pyqt-46
        if __version__ < LooseVersion("4.6"):
            raise ImportError("PyQt<4.6 is not supported")
        QtCore.Signal = QtCore.pyqtSignal
        QtCore.Slot = QtCore.pyqtSlot
        QtCore.Property = QtCore.pyqtProperty
        _isdeleted = sip.isdeleted
        _getSaveFileName = QtGui.QFileDialog.getSaveFileNameAndFilter

    if QT_API == QT_API_PYQTv2:
        _setup_pyqt4_internal(api=2)
    elif QT_API == QT_API_PYSIDE:
        from PySide import QtCore, QtGui, __version__, __version_info__
        import shiboken
        # PySide 1.0.3 fixed the following:
        # https://srinikom.github.io/pyside-bz-archive/809.html
        if __version_info__ < (1, 0, 3):
            raise ImportError("PySide<1.0.3 is not supported")
        def _isdeleted(obj): return not shiboken.isValid(obj)
        _getSaveFileName = QtGui.QFileDialog.getSaveFileName
    elif QT_API == QT_API_PYQT:
        _setup_pyqt4_internal(api=1)
    else:
        raise ValueError("Unexpected value for the 'backend.qt4' rcparam")
    QtWidgets = QtGui

    @_api.deprecated("3.3", alternative="QtCore.qVersion()")
    def is_pyqt5():
        return False


if QT_API in [QT_API_PYQT5, QT_API_PYSIDE2]:
    _setup_pyqt5()
elif QT_API in [QT_API_PYQTv2, QT_API_PYSIDE, QT_API_PYQT]:
    _setup_pyqt4()
elif QT_API is None:  # See above re: dict.__getitem__.
    if dict.__getitem__(mpl.rcParams, "backend") == "Qt4Agg":
        _candidates = [(_setup_pyqt4, QT_API_PYQTv2),
                       (_setup_pyqt4, QT_API_PYSIDE),
                       (_setup_pyqt4, QT_API_PYQT),
                       (_setup_pyqt5, QT_API_PYQT5),
                       (_setup_pyqt5, QT_API_PYSIDE2)]
    else:
        _candidates = [(_setup_pyqt5, QT_API_PYQT5),
                       (_setup_pyqt5, QT_API_PYSIDE2),
                       (_setup_pyqt4, QT_API_PYQTv2),
                       (_setup_pyqt4, QT_API_PYSIDE),
                       (_setup_pyqt4, QT_API_PYQT)]
    for _setup, QT_API in _candidates:
        try:
            _setup()
        except ImportError:
            continue
        break
    else:
        raise ImportError("Failed to import any qt binding")
else:  # We should not get there.
    raise AssertionError("Unexpected QT_API: {}".format(QT_API))


# Fixes issues with Big Sur
# https://bugreports.qt.io/browse/QTBUG-87014, fixed in qt 5.15.2
if (sys.platform == 'darwin' and
        LooseVersion(platform.mac_ver()[0]) >= LooseVersion("10.16") and
        LooseVersion(QtCore.qVersion()) < LooseVersion("5.15.2") and
        "QT_MAC_WANTS_LAYER" not in os.environ):
    os.environ["QT_MAC_WANTS_LAYER"] = "1"


# These globals are only defined for backcompatibility purposes.
ETS = dict(pyqt=(QT_API_PYQTv2, 4), pyside=(QT_API_PYSIDE, 4),
           pyqt5=(QT_API_PYQT5, 5), pyside2=(QT_API_PYSIDE2, 5))

QT_RC_MAJOR_VERSION = int(QtCore.qVersion().split(".")[0])

if QT_RC_MAJOR_VERSION == 4:
    _api.warn_deprecated("3.3", name="support for Qt4")


def _devicePixelRatioF(obj):
    """
    Return obj.devicePixelRatioF() with graceful fallback for older Qt.

    This can be replaced by the direct call when we require Qt>=5.6.
    """
    try:
        # Not available on Qt<5.6
        return obj.devicePixelRatioF() or 1
    except AttributeError:
        pass
    try:
        # Not available on Qt4 or some older Qt5.
        # self.devicePixelRatio() returns 0 in rare cases
        return obj.devicePixelRatio() or 1
    except AttributeError:
        return 1


def _setDevicePixelRatio(obj, val):
    """
    Call obj.setDevicePixelRatio(val) with graceful fallback for older Qt.

    This can be replaced by the direct call when we require Qt>=5.6.
    """
    if hasattr(obj, 'setDevicePixelRatio'):
        # Not available on Qt4 or some older Qt5.
        obj.setDevicePixelRatio(val)
