import os
import subprocess
import sys

import pytest

_test_timeout = 10  # Empirically, 1s is not enough on CI.

# NOTE: TkAgg tests seem to have interactions between tests,
# So isolate each test in a subprocess. See GH#18261


@pytest.mark.backend('TkAgg', skip_on_importerror=True)
def test_blit():
    script = """
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends import _tkagg
def evil_blit(photoimage, aggimage, offsets, bboxptr):
    data = np.asarray(aggimage)
    height, width = data.shape[:2]
    dataptr = (height, width, data.ctypes.data)
    _tkagg.blit(
        photoimage.tk.interpaddr(), str(photoimage), dataptr, offsets,
        bboxptr)

fig, ax = plt.subplots()
bad_boxes = ((-1, 2, 0, 2),
             (2, 0, 0, 2),
             (1, 6, 0, 2),
             (0, 2, -1, 2),
             (0, 2, 2, 0),
             (0, 2, 1, 6))
for bad_box in bad_boxes:
    try:
        evil_blit(fig.canvas._tkphoto,
                  np.ones((4, 4, 4)),
                  (0, 1, 2, 3),
                  bad_box)
    except ValueError:
        print("success")
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ,
                 "MPLBACKEND": "TkAgg",
                 "SOURCE_DATE_EPOCH": "0"},
            timeout=_test_timeout,
            stdout=subprocess.PIPE,
            check=True,
            universal_newlines=True,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Subprocess timed out")
    except subprocess.CalledProcessError:
        pytest.fail("Likely regression on out-of-bounds data access"
                    " in _tkagg.cpp")
    else:
        print(proc.stdout)
        assert proc.stdout.count("success") == 6  # len(bad_boxes)


@pytest.mark.backend('TkAgg', skip_on_importerror=True)
def test_figuremanager_preserves_host_mainloop():
    script = """
import tkinter
import matplotlib.pyplot as plt
success = False

def do_plot():
    plt.figure()
    plt.plot([1, 2], [3, 5])
    plt.close()
    root.after(0, legitimate_quit)

def legitimate_quit():
    root.quit()
    global success
    success = True

root = tkinter.Tk()
root.after(0, do_plot)
root.mainloop()

if success:
    print("success")
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ,
                 "MPLBACKEND": "TkAgg",
                 "SOURCE_DATE_EPOCH": "0"},
            timeout=_test_timeout,
            stdout=subprocess.PIPE,
            check=True,
            universal_newlines=True,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Subprocess timed out")
    except subprocess.CalledProcessError:
        pytest.fail("Subprocess failed to test intended behavior")
    else:
        assert proc.stdout.count("success") == 1


@pytest.mark.backend('TkAgg', skip_on_importerror=True)
@pytest.mark.flaky(reruns=3)
def test_figuremanager_cleans_own_mainloop():
    script = '''
import tkinter
import time
import matplotlib.pyplot as plt
import threading
from matplotlib.cbook import _get_running_interactive_framework

root = tkinter.Tk()
plt.plot([1, 2, 3], [1, 2, 5])

def target():
    while not 'tk' == _get_running_interactive_framework():
        time.sleep(.01)
    plt.close()
    if show_finished_event.wait():
        print('success')

show_finished_event = threading.Event()
thread = threading.Thread(target=target, daemon=True)
thread.start()
plt.show(block=True)  # testing if this function hangs
show_finished_event.set()
thread.join()

'''
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ,
                 "MPLBACKEND": "TkAgg",
                 "SOURCE_DATE_EPOCH": "0"},
            timeout=_test_timeout,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            check=True
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Most likely plot.show(block=True) hung")
    except subprocess.CalledProcessError:
        pytest.fail("Subprocess failed to test intended behavior")
    assert proc.stdout.count("success") == 1


@pytest.mark.backend('TkAgg', skip_on_importerror=True)
@pytest.mark.flaky(reruns=3)
def test_never_update():
    script = """
import tkinter
del tkinter.Misc.update
del tkinter.Misc.update_idletasks

import matplotlib.pyplot as plt
fig = plt.figure()
plt.show(block=False)

# regression test on FigureCanvasTkAgg
plt.draw()
# regression test on NavigationToolbar2Tk
fig.canvas.toolbar.configure_subplots()

# check for update() or update_idletasks() in the event queue
# functionally equivalent to tkinter.Misc.update
# must pause >= 1 ms to process tcl idle events plus
# extra time to avoid flaky tests on slow systems
plt.pause(0.1)

# regression test on FigureCanvasTk filter_destroy callback
plt.close(fig)
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ,
                 "MPLBACKEND": "TkAgg",
                 "SOURCE_DATE_EPOCH": "0"},
            timeout=_test_timeout,
            capture_output=True,
            universal_newlines=True,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Subprocess timed out")
    else:
        # test framework doesn't see tkinter callback exceptions normally
        # see tkinter.Misc.report_callback_exception
        assert "Exception in Tkinter callback" not in proc.stderr
        # make sure we can see other issues
        print(proc.stderr, file=sys.stderr)
        # Checking return code late so the Tkinter assertion happens first
        if proc.returncode:
            pytest.fail("Subprocess failed to test intended behavior")


@pytest.mark.backend('TkAgg', skip_on_importerror=True)
def test_missing_back_button():
    script = """
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
class Toolbar(NavigationToolbar2Tk):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom')]

fig = plt.figure()
print("setup complete")
# this should not raise
Toolbar(fig.canvas, fig.canvas.manager.window)
print("success")
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ,
                 "MPLBACKEND": "TkAgg",
                 "SOURCE_DATE_EPOCH": "0"},
            timeout=_test_timeout,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Subprocess timed out")
    else:
        assert proc.stdout.count("setup complete") == 1
        assert proc.stdout.count("success") == 1
        # Checking return code late so the stdout assertions happen first
        if proc.returncode:
            pytest.fail("Subprocess failed to test intended behavior")
