"""Long-recording behaviour: memory stays flat and failures stay at the same indices.

These tests exist because a 25 661-frame server run reached ~3000 frames and failed
with what looked like a decoder memory problem. Measurements on the decode path,
the radiometry and both writers show a flat RSS; what these tests protect is that
property (and the determinism of per-frame failures) from regressing.
"""

from __future__ import annotations

import gc
import json
import resource
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from flir_csq.container import CsqReader, scan_csq
from flir_csq.diagnostics import check_frames
from flir_csq.synthetic import make_streaming_csq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _rss_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def _peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc/self/status")
def test_decoder_memory_is_flat_over_thousands_of_frames(tmp_path):
    path = tmp_path / "many.csq"
    make_streaming_csq(path, n_frames=1200, height=96, width=128, noise_c=1.0)
    index = scan_csq(path)
    del index
    gc.collect()
    base = _rss_mb()
    samples = []
    with CsqReader(scan_csq(path)) as reader:
        for i in range(len(reader)):
            reader.read_frame(i)
            if i % 300 == 299:
                gc.collect()
                samples.append(_rss_mb())
    growth = samples[-1] - base
    print(f"RSS after every 300 frames: {[round(s, 1) for s in samples]}, "
          f"growth {growth:+.1f} MB, peak {_peak_mb():.0f} MB")
    assert growth < 25.0, f"decode path grew {growth:.1f} MB over 1200 frames"
    assert samples[-1] - samples[1] < 15.0, "growth continues after warm-up"


CONVERT_SCRIPT = textwrap.dedent(
    """
    import json, resource, sys
    sys.path.insert(0, {src!r})
    from flir_csq.converter import ConversionOptions, convert
    out = convert(ConversionOptions(input_path={path!r}, output_dir={out!r},
                                    formats=("h5",), save_raw=False,
                                    save_metadata=False))
    print(json.dumps({{"frames": out.frames_written,
                       "peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}}))
    """
)


def _convert_peak(frames: int, tmp_path: Path) -> float:
    path = tmp_path / f"long_{frames}.csq"
    make_streaming_csq(path, n_frames=frames, height=72, width=96)
    script = CONVERT_SCRIPT.format(src=str(SRC), path=str(path), out=str(tmp_path / f"o{frames}"))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=900, check=False)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["frames"] == frames
    return payload["peak_mb"]


def test_conversion_memory_does_not_grow_with_frame_count(tmp_path):
    small = _convert_peak(200, tmp_path)
    large = _convert_peak(3000, tmp_path)
    print(f"peak RSS: 200 frames {small:.1f} MB, 3000 frames {large:.1f} MB")
    assert large - small < 60.0, (
        f"peak RSS grew {large - small:.1f} MB between 200 and 3000 frames - "
        "the conversion loop is retaining frames"
    )


def test_failures_stay_at_the_same_frame_indices(tmp_path):
    path = tmp_path / "flaky.csq"
    bad = (1234, 2499)
    make_streaming_csq(path, n_frames=2600, height=48, width=64, corrupt_frames=bad)
    first = check_frames(path)
    second = check_frames(path)
    assert first["failed_indices"] == list(bad)
    assert second["failed_indices"] == first["failed_indices"]
    assert first["decoded"] == 2598
    for failure in first["failures"]:
        assert failure["payload_sha256"] == second["failures"][first["failures"].index(failure)][
            "payload_sha256"], "a frame's payload must be identical between runs"


def test_long_mat73_conversion_streams(tmp_path):
    path = tmp_path / "long_mat.csq"
    make_streaming_csq(path, n_frames=1500, height=72, width=96)
    script = textwrap.dedent(
        f"""
        import json, resource, sys
        sys.path.insert(0, {str(SRC)!r})
        from flir_csq.converter import ConversionOptions, convert
        out = convert(ConversionOptions(input_path={str(path)!r},
                                        output_dir={str(tmp_path / "mat")!r},
                                        formats=("mat",), mat_version="7.3",
                                        save_raw=False, save_metadata=False))
        print(json.dumps({{"frames": out.frames_written,
                           "peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}}))
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=900, check=False)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["frames"] == 1500
    print(f"MAT v7.3, 1500 frames: peak {payload['peak_mb']:.0f} MB")
    assert payload["peak_mb"] < 400.0, "MAT v7.3 writing is not streaming"


def test_batch_does_not_accumulate_across_files(tmp_path):

    inp = tmp_path / "batch_in"
    for n in range(3):
        make_streaming_csq(inp / f"rec_{n}.csq", n_frames=300, height=72, width=96)
    script = textwrap.dedent(
        f"""
        import json, resource, sys
        sys.path.insert(0, {str(SRC)!r})
        from flir_csq.batch import BatchOptions, run_batch
        result = run_batch(BatchOptions(input_dir={str(inp)!r},
                                        output_dir={str(tmp_path / 'batch_out')!r},
                                        formats=("h5",), save_raw=False))
        print(json.dumps({{"converted": result.converted,
                           "peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}}))
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=900, check=False)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["converted"] == 3
    print(f"batch of 3 recordings: peak {payload['peak_mb']:.0f} MB")
    assert payload["peak_mb"] < 400.0
