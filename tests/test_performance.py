"""Performance and memory behaviour on a longer recording.

The point is not absolute speed (that depends on the machine) but that peak RSS
does not grow with the number of frames: the converter must stream.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

SCRIPT = textwrap.dedent(
    """
    import json, resource, sys
    sys.path.insert(0, {src!r})
    from flir_csq.converter import ConversionOptions, convert
    out = convert(ConversionOptions(input_path={path!r}, output_dir={out!r},
                                    formats=("h5",), save_metadata=False))
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(json.dumps({{"frames": out.frames_written, "peak_mb": peak_mb}}))
    """
)


def _peak_rss(frames: int, tmp_path: Path) -> float:
    from flir_csq.synthetic import make_synthetic_csq

    path = tmp_path / f"long_{frames}.csq"
    make_synthetic_csq(path, n_frames=frames, height=64, width=48, compression="jpegls")
    out = tmp_path / f"out_{frames}"
    script = SCRIPT.format(src=str(SRC), path=str(path), out=str(out))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False,
                          timeout=600)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["frames"] == frames
    return payload["peak_mb"]


def test_peak_memory_does_not_grow_with_frame_count(tmp_path):
    small = _peak_rss(40, tmp_path)
    large = _peak_rss(600, tmp_path)
    growth = large - small
    print(f"peak RSS: 40 frames {small:.1f} MB, 600 frames {large:.1f} MB")
    assert growth < 60.0, (
        f"peak memory grew by {growth:.1f} MB between 40 and 600 frames - the converter is "
        "not streaming"
    )


def test_conversion_speed_is_reasonable(jpegls_recording, tmp_path):
    from flir_csq.converter import ConversionOptions, convert

    result = convert(ConversionOptions(input_path=jpegls_recording["path"], output_dir=tmp_path,
                                       formats=("h5",), save_metadata=False))
    per_frame = result.duration_s / max(result.frames_written, 1)
    print(f"{result.duration_s:.2f} s for {result.frames_written} frames "
          f"({per_frame * 1000:.1f} ms/frame)")
    assert per_frame < 0.5
