"""Tests for tinypages build using sphinx extensions."""

import filecmp
import os
from pathlib import Path
from subprocess import Popen, PIPE
import sys

import pytest


pytest.importorskip('sphinx')


def test_tinypages(tmpdir):
    tmp_path = Path(tmpdir)
    html_dir = tmp_path / 'html'
    doctree_dir = tmp_path / 'doctrees'
    # Build the pages with warnings turned into errors
    cmd = [sys.executable, '-msphinx', '-W', '-b', 'html',
           '-d', str(doctree_dir),
           str(Path(__file__).parent / 'tinypages'), str(html_dir)]
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True,
                 env={**os.environ, "MPLBACKEND": ""})
    out, err = proc.communicate()

    assert proc.returncode == 0, \
        f"sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n"
    if err:
        pytest.fail(f"sphinx build emitted the following warnings:\n{err}")

    assert html_dir.is_dir()

    def plot_file(num):
        return html_dir / f'some_plots-{num}.png'

    range_10, range_6, range_4 = [plot_file(i) for i in range(1, 4)]
    # Plot 5 is range(6) plot
    assert filecmp.cmp(range_6, plot_file(5))
    # Plot 7 is range(4) plot
    assert filecmp.cmp(range_4, plot_file(7))
    # Plot 11 is range(10) plot
    assert filecmp.cmp(range_10, plot_file(11))
    # Plot 12 uses the old range(10) figure and the new range(6) figure
    assert filecmp.cmp(range_10, plot_file('12_00'))
    assert filecmp.cmp(range_6, plot_file('12_01'))
    # Plot 13 shows close-figs in action
    assert filecmp.cmp(range_4, plot_file(13))
    # Plot 14 has included source
    html_contents = (html_dir / 'some_plots.html').read_bytes()
    assert b'# Only a comment' in html_contents
    # check plot defined in external file.
    assert filecmp.cmp(range_4, html_dir / 'range4.png')
    assert filecmp.cmp(range_6, html_dir / 'range6.png')
    # check if figure caption made it into html file
    assert b'This is the caption for plot 15.' in html_contents
    # check if figure caption using :caption: made it into html file
    assert b'Plot 17 uses the caption option.' in html_contents
    # check if figure caption made it into html file
    assert b'This is the caption for plot 18.' in html_contents
    # check if the custom classes made it into the html file
    assert b'plot-directive my-class my-other-class' in html_contents
    # check that the multi-image caption is applied twice
    assert html_contents.count(b'This caption applies to both plots.') == 2
