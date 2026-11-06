#!/usr/bin/env python3
"""Convert a FLIR T-series ``.csq`` recording into temperature matrices.

Usage::

    python convert_csq.py experiment_001.csq
    python convert_csq.py experiment_001.csq --output results --format h5,csv --save-raw
    python convert_csq.py experiment_001.csq --inspect
    python convert_csq.py --demo /tmp/demo_t540.csq

Run ``python convert_csq.py --help`` for every option.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running straight from a clone, without `pip install -e .`.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from flir_csq.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
