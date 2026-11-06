"""Frame-level diagnostics: which frames of a recording can actually be decoded?

A conversion stops at the first unreadable frame, which is correct for science but
useless for diagnosis. This scans every frame, records the exact exception per
frame, and reports whether failures are reproducible at the same indices - the
question that matters when a long recording fails "somewhere around frame 3000".
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Callable

from .container import CsqReader, scan_csq
from .errors import CsqError


def check_frames(path: str | Path, *, start: int = 0, stop: int | None = None,
                 stride: int = 1, dump_dir: str | Path | None = None,
                 allow_fallback: bool = False,
                 progress: Callable[[float, str, int, int], None] | None = None) -> dict[str, Any]:
    """Decode every selected frame without converting or writing anything."""
    p = Path(path)
    index = scan_csq(p)
    complete = [i for i, extent in enumerate(index.frames) if not extent.truncated]
    stop = min(stop if stop is not None else len(complete), len(complete))
    frames = list(range(start, stop, max(1, stride)))
    reader = CsqReader(index, allow_fallback=allow_fallback)
    failures: list[dict[str, Any]] = []
    dump_root = Path(dump_dir) if dump_dir is not None else None
    if dump_root is not None:
        dump_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    ok = 0
    try:
        for n, frame_index in enumerate(frames, start=1):
            try:
                reader.read_frame(frame_index)
                ok += 1
            except CsqError as exc:
                payload = reader.raw_payload(frame_index)
                entry: dict[str, Any] = {
                    "frame_index": frame_index,
                    "error_type": type(exc).__name__,
                    "error": exc.message,
                    "hint": exc.hint,
                    "payload_bytes": len(payload) if payload is not None else None,
                    "payload_sha256": hashlib.sha256(payload).hexdigest() if payload else None,
                    "cause": (f"{type(exc.cause).__name__}: {exc.cause}"
                              if getattr(exc, "cause", None) is not None else None),
                }
                attempts = getattr(exc, "attempts", None)
                if attempts:
                    entry["attempts"] = [{"codec": name, "error": text}
                                         for name, text in attempts]
                if dump_root is not None and payload is not None:
                    target = dump_root / f"{p.stem}_frame_{frame_index:06d}.jls"
                    target.write_bytes(payload)
                    entry["payload_file"] = str(target)
                failures.append(entry)
            if progress and (n % 50 == 0 or n == len(frames)):
                progress(n / max(len(frames), 1), f"checked {n}/{len(frames)}",
                         n, len(frames))
    finally:
        reader.close()

    elapsed = time.perf_counter() - started
    indices = [f["frame_index"] for f in failures]
    return {
        "path": str(p.resolve()),
        "n_frames_in_file": index.n_frames,
        "checked": len(frames),
        "decoded": ok,
        "failed": len(failures),
        "failed_indices": indices,
        "first_failure": indices[0] if indices else None,
        "last_failure": indices[-1] if indices else None,
        "all_decoded": not failures,
        "failures": failures[:100],
        "truncated_tail": any(f.truncated for f in index.frames),
        "elapsed_s": round(elapsed, 2),
        "frames_per_second": round(len(frames) / elapsed, 1) if elapsed > 0 else None,
        "container_warnings": index.warnings,
    }
