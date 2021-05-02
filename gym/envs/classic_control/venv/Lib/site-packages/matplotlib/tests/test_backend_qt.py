import copy
import signal
from unittest import mock

import matplotlib
from matplotlib import pyplot as plt
from matplotlib._pylab_helpers import Gcf

import pytest


try:
    from matplotlib.backends.qt_compat import QtGui
except ImportError:
    pytestmark = pytest.mark.skip('No usable Qt5 bindings')


@pytest.fixture
def qt_core(request):
    backend, = request.node.get_closest_marker('backend').args
    qt_compat = pytest.importorskip('matplotlib.backends.qt_compat')
    QtCore = qt_compat.QtCore

    if backend == 'Qt4Agg':
        try:
            py_qt_ver = int(QtCore.PYQT_VERSION_STR.split('.')[0])
        except AttributeError:
            py_qt_ver = QtCore.__version_info__[0]
        if py_qt_ver != 4:
            pytest.skip('Qt4 is not available')

    return QtCore


@pytest.mark.parametrize('backend', [
    # Note: the value is irrelevant; the important part is the marker.
    pytest.param(
        'Qt4Agg',
        marks=pytest.mark.backend('Qt4Agg', skip_on_importerror=True)),
    pytest.param(
        'Qt5Agg',
        marks=pytest.mark.backend('Qt5Agg', skip_on_importerror=True)),
])
def test_fig_close(backend):
    # save the state of Gcf.figs
    init_figs = copy.copy(Gcf.figs)

    # make a figure using pyplot interface
    fig = plt.figure()

    # simulate user clicking the close button by reaching in
    # and calling close on the underlying Qt object
    fig.canvas.manager.window.close()

    # assert that we have removed the reference to the FigureManager
    # that got added by plt.figure()
    assert init_figs == Gcf.figs


@pytest.mark.backend('Qt5Agg', skip_on_importerror=True)
def test_fig_signals(qt_core):
    # Create a figure
    plt.figure()

    # Access signals
    event_loop_signal = None

    # Callback to fire during event loop: save SIGINT handler, then exit
    def fire_signal_and_quit():
        # Save event loop signal
        nonlocal event_loop_signal
        event_loop_signal = signal.getsignal(signal.SIGINT)

        # Request event loop exit
        qt_core.QCoreApplication.exit()

    # Timer to exit event loop
    qt_core.QTimer.singleShot(0, fire_signal_and_quit)

    # Save original SIGINT handler
    original_signal = signal.getsignal(signal.SIGINT)

    # Use our own SIGINT handler to be 100% sure this is working
    def CustomHandler(signum, frame):
        pass

    signal.signal(signal.SIGINT, CustomHandler)

    # mainloop() sets SIGINT, starts Qt event loop (which triggers timer and
    # exits) and then mainloop() resets SIGINT
    matplotlib.backends.backend_qt5._BackendQT5.mainloop()

    # Assert: signal handler during loop execution is signal.SIG_DFL
    assert event_loop_signal == signal.SIG_DFL

    # Assert: current signal handler is the same as the one we set before
    assert CustomHandler == signal.getsignal(signal.SIGINT)

    # Reset SIGINT handler to what it was before the test
    signal.signal(signal.SIGINT, original_signal)


@pytest.mark.parametrize(
    'qt_key, qt_mods, answer',
    [
        ('Key_A', ['ShiftModifier'], 'A'),
        ('Key_A', [], 'a'),
        ('Key_A', ['ControlModifier'], 'ctrl+a'),
        ('Key_Aacute', ['ShiftModifier'],
         '\N{LATIN CAPITAL LETTER A WITH ACUTE}'),
        ('Key_Aacute', [],
         '\N{LATIN SMALL LETTER A WITH ACUTE}'),
        ('Key_Control', ['AltModifier'], 'alt+control'),
        ('Key_Alt', ['ControlModifier'], 'ctrl+alt'),
        ('Key_Aacute', ['ControlModifier', 'AltModifier', 'MetaModifier'],
         'ctrl+alt+super+\N{LATIN SMALL LETTER A WITH ACUTE}'),
        ('Key_Play', [], None),
        ('Key_Backspace', [], 'backspace'),
        ('Key_Backspace', ['ControlModifier'], 'ctrl+backspace'),
    ],
    ids=[
        'shift',
        'lower',
        'control',
        'unicode_upper',
        'unicode_lower',
        'alt_control',
        'control_alt',
        'modifier_order',
        'non_unicode_key',
        'backspace',
        'backspace_mod',
    ]
)
@pytest.mark.parametrize('backend', [
    # Note: the value is irrelevant; the important part is the marker.
    pytest.param(
        'Qt4Agg',
        marks=pytest.mark.backend('Qt4Agg', skip_on_importerror=True)),
    pytest.param(
        'Qt5Agg',
        marks=pytest.mark.backend('Qt5Agg', skip_on_importerror=True)),
])
def test_correct_key(backend, qt_core, qt_key, qt_mods, answer):
    """
    Make a figure.
    Send a key_press_event event (using non-public, qtX backend specific api).
    Catch the event.
    Assert sent and caught keys are the same.
    """
    qt_mod = qt_core.Qt.NoModifier
    for mod in qt_mods:
        qt_mod |= getattr(qt_core.Qt, mod)

    class _Event:
        def isAutoRepeat(self): return False
        def key(self): return getattr(qt_core.Qt, qt_key)
        def modifiers(self): return qt_mod

    def on_key_press(event):
        assert event.key == answer

    qt_canvas = plt.figure().canvas
    qt_canvas.mpl_connect('key_press_event', on_key_press)
    qt_canvas.keyPressEvent(_Event())


@pytest.mark.backend('Qt5Agg', skip_on_importerror=True)
def test_pixel_ratio_change():
    """
    Make sure that if the pixel ratio changes, the figure dpi changes but the
    widget remains the same physical size.
    """

    prop = 'matplotlib.backends.backend_qt5.FigureCanvasQT.devicePixelRatioF'
    with mock.patch(prop) as p:
        p.return_value = 3

        fig = plt.figure(figsize=(5, 2), dpi=120)
        qt_canvas = fig.canvas
        qt_canvas.show()

        def set_pixel_ratio(ratio):
            p.return_value = ratio
            # Make sure the mocking worked
            assert qt_canvas._dpi_ratio == ratio

            # The value here doesn't matter, as we can't mock the C++ QScreen
            # object, but can override the functional wrapper around it.
            # Emitting this event is simply to trigger the DPI change handler
            # in Matplotlib in the same manner that it would occur normally.
            screen.logicalDotsPerInchChanged.emit(96)

            qt_canvas.draw()
            qt_canvas.flush_events()

        qt_canvas.manager.show()
        size = qt_canvas.size()
        screen = qt_canvas.window().windowHandle().screen()
        set_pixel_ratio(3)

        # The DPI and the renderer width/height change
        assert fig.dpi == 360
        assert qt_canvas.renderer.width == 1800
        assert qt_canvas.renderer.height == 720

        # The actual widget size and figure physical size don't change
        assert size.width() == 600
        assert size.height() == 240
        assert qt_canvas.get_width_height() == (600, 240)
        assert (fig.get_size_inches() == (5, 2)).all()

        set_pixel_ratio(2)

        # The DPI and the renderer width/height change
        assert fig.dpi == 240
        assert qt_canvas.renderer.width == 1200
        assert qt_canvas.renderer.height == 480

        # The actual widget size and figure physical size don't change
        assert size.width() == 600
        assert size.height() == 240
        assert qt_canvas.get_width_height() == (600, 240)
        assert (fig.get_size_inches() == (5, 2)).all()

        set_pixel_ratio(1.5)

        # The DPI and the renderer width/height change
        assert fig.dpi == 180
        assert qt_canvas.renderer.width == 900
        assert qt_canvas.renderer.height == 360

        # The actual widget size and figure physical size don't change
        assert size.width() == 600
        assert size.height() == 240
        assert qt_canvas.get_width_height() == (600, 240)
        assert (fig.get_size_inches() == (5, 2)).all()


@pytest.mark.backend('Qt5Agg', skip_on_importerror=True)
def test_subplottool():
    fig, ax = plt.subplots()
    with mock.patch(
            "matplotlib.backends.backend_qt5.SubplotToolQt.exec_",
            lambda self: None):
        fig.canvas.manager.toolbar.configure_subplots()


@pytest.mark.backend('Qt5Agg', skip_on_importerror=True)
def test_figureoptions():
    fig, ax = plt.subplots()
    ax.plot([1, 2])
    ax.imshow([[1]])
    ax.scatter(range(3), range(3), c=range(3))
    with mock.patch(
            "matplotlib.backends.qt_editor._formlayout.FormDialog.exec_",
            lambda self: None):
        fig.canvas.manager.toolbar.edit_parameters()


@pytest.mark.backend('Qt5Agg', skip_on_importerror=True)
def test_double_resize():
    # Check that resizing a figure twice keeps the same window size
    fig, ax = plt.subplots()
    fig.canvas.draw()
    window = fig.canvas.manager.window

    w, h = 3, 2
    fig.set_size_inches(w, h)
    assert fig.canvas.width() == w * matplotlib.rcParams['figure.dpi']
    assert fig.canvas.height() == h * matplotlib.rcParams['figure.dpi']

    old_width = window.width()
    old_height = window.height()

    fig.set_size_inches(w, h)
    assert window.width() == old_width
    assert window.height() == old_height


@pytest.mark.backend('Qt5Agg', skip_on_importerror=True)
def test_canvas_reinit():
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

    called = False

    def crashing_callback(fig, stale):
        nonlocal called
        fig.canvas.draw_idle()
        called = True

    fig, ax = plt.subplots()
    fig.stale_callback = crashing_callback
    # this should not raise
    canvas = FigureCanvasQTAgg(fig)
    fig.stale = True
    assert called
