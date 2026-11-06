"""RSS measurement helpers, so memory behaviour can be observed instead of guessed.

``MaxRSS`` from the kernel is the number SLURM reports, so that is what is
recorded in the conversion metadata and written to the log when the user asks for
memory logging.
"""

from __future__ import annotations

import resource
import sys
from typing import Any


def current_rss_mb() -> float | None:
    """Resident set size right now (Linux ``/proc/self/status``)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def peak_rss_mb() -> float | None:
    """Peak resident set size for this process (what SLURM reports as MaxRSS)."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):  # pragma: no cover - non-POSIX
        return None
    # Linux reports kilobytes, macOS bytes.
    return usage / (1024.0 * 1024.0) if sys.platform == "darwin" else usage / 1024.0


def memory_snapshot() -> dict[str, Any]:
    return {"rss_mb": _round(current_rss_mb()), "peak_rss_mb": _round(peak_rss_mb())}


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 1)
