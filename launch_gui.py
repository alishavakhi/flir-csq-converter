#!/usr/bin/env python3
"""Launch the FLIR CSQ → Temperature desktop application (PySide6).

Usage::

    python launch_gui.py
    python launch_gui.py --self-test     # headless construction + demo conversion
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> int:
    try:
        from flir_csq.gui import main as gui_main
    except ImportError as exc:  # pragma: no cover - depends on the environment
        print(
            "PySide6 is required for the graphical interface.\n"
            "  python3 -m venv .venv && source .venv/bin/activate\n"
            "  pip install -r requirements.txt\n"
            f"(import error: {exc})",
            file=sys.stderr,
        )
        return 2
    return gui_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
