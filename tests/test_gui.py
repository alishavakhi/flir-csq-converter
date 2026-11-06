"""GUI smoke test: the PySide6 window builds and converts a recording headless."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("PySide6")


def test_gui_self_test_runs_headless():
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT / "src"))
    proc = subprocess.run([sys.executable, str(ROOT / "launch_gui.py"), "--self-test"],
                          capture_output=True, text=True, check=False, env=env, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "self-test" in proc.stdout


def test_gui_module_imports_and_builds_window():
    from flir_csq.gui import MainWindow

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert "FLIR CSQ" in window.windowTitle()
    assert window.convert_button.isEnabled() is False  # nothing selected yet
    window.close()
    app.processEvents()
