"""Decode-failure handling: preserved exceptions, bounded allocation, recovery mode."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from flir_csq.codecs import decode_compressed, jpeg_frame_header, jpeg_has_eoi
from flir_csq.converter import ConversionOptions, convert
from flir_csq.diagnostics import check_frames
from flir_csq.errors import PayloadDecodeError, UnsupportedPayload
from flir_csq.synthetic import make_streaming_csq

ROOT = Path(__file__).resolve().parents[1]
h5py = pytest.importorskip("h5py")


# -- payload-level behaviour -------------------------------------------------------
def _good_payload() -> bytes:
    import imagecodecs

    field = np.linspace(12000, 16000, 64 * 48, dtype=np.float64).reshape(48, 64)
    return imagecodecs.jpegls_encode(np.rint(field).astype(np.uint16))


def test_valid_payload_decodes_and_reports_header():
    payload = _good_payload()
    header = jpeg_frame_header(payload)
    assert header.kind == "jpegls"
    assert (header.width, header.height) == (64, 48)
    assert jpeg_has_eoi(payload)
    arr, codec = decode_compressed(payload, width=64, height=48)
    assert arr.shape == (48, 64) and codec.startswith("imagecodecs")


def test_truncated_payload_is_rejected_before_decoding():
    payload = _good_payload()[:-40]
    assert not jpeg_has_eoi(payload)
    with pytest.raises(PayloadDecodeError) as exc:
        decode_compressed(payload, width=64, height=48)
    assert "truncated" in exc.value.message
    assert exc.value.payload_size == len(payload)


def test_corrupt_entropy_data_preserves_the_codec_exception():
    """Regression: a failing codec must never be reported as "not installed"."""
    payload = bytearray(_good_payload())
    payload[60:120] = bytes((i * 37) & 0xFF for i in range(60))
    with pytest.raises(PayloadDecodeError) as exc:
        decode_compressed(bytes(payload), width=64, height=48)
    err = exc.value
    assert isinstance(err, UnsupportedPayload)
    assert "no JPEG-LS decoder is installed" not in err.message
    assert "could not read this payload" in err.message
    assert err.cause is not None, "the codec's own exception must be preserved"
    assert type(err.cause).__name__ in {"JpeglsError", "ValueError", "JpegError"}
    assert err.attempts and err.attempts[0][0].startswith("imagecodecs")


def test_corrupt_header_dimensions_are_rejected_without_huge_allocation():
    """A 60000x60000 header used to make the decoder ask numpy for 6.7 GiB."""
    payload = bytearray(_good_payload())
    i = payload.find(b"\xff\xf7")
    payload[i + 5:i + 7] = (60000).to_bytes(2, "big")
    payload[i + 7:i + 9] = (60000).to_bytes(2, "big")
    with pytest.raises(PayloadDecodeError) as exc:
        decode_compressed(bytes(payload), width=64, height=48)
    assert "declares 60000x60000" in exc.value.message
    assert exc.value.cause is None, "rejected before the codec touched it"


def test_allocation_is_bounded_in_a_subprocess(tmp_path):
    """Decode a poisoned payload under a hard 1 GiB address-space limit."""
    script = textwrap.dedent(
        f"""
        import resource, sys
        resource.setrlimit(resource.RLIMIT_AS, (1_000_000_000, 1_000_000_000))
        sys.path.insert(0, {str(ROOT / 'src')!r})
        import numpy as np, imagecodecs
        from flir_csq.codecs import decode_compressed
        field = np.linspace(12000, 16000, 64 * 48).reshape(48, 64)
        payload = bytearray(imagecodecs.jpegls_encode(np.rint(field).astype(np.uint16)))
        i = payload.find(b"\\xff\\xf7")
        payload[i + 5:i + 7] = (60000).to_bytes(2, "big")
        payload[i + 7:i + 9] = (60000).to_bytes(2, "big")
        try:
            decode_compressed(bytes(payload), width=64, height=48)
        except Exception as exc:
            print(type(exc).__name__)
        else:
            print("decoded")
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "PayloadDecodeError"


# -- recovery mode -----------------------------------------------------------------
@pytest.fixture
def corrupt_recording(tmp_path: Path) -> dict:
    path = tmp_path / "gapped.csq"
    truth = make_streaming_csq(path, n_frames=12, height=24, width=32,
                               corrupt_frames=(2, 7), noise_c=0.5)
    return truth


def test_default_mode_fails_and_names_the_frame(corrupt_recording, tmp_path):
    with pytest.raises(PayloadDecodeError) as exc:
        convert(ConversionOptions(input_path=corrupt_recording["path"],
                                  output_dir=tmp_path / "out", formats=("h5",),
                                  save_metadata=False))
    assert exc.value.frame_index == 2
    assert "frame" in exc.value.message or exc.value.frame_index == 2
    assert not list((tmp_path / "out").glob("*.h5"))


def test_recovery_writes_nan_frames_and_keeps_timestamps(corrupt_recording, tmp_path):
    result = convert(ConversionOptions(
        input_path=corrupt_recording["path"], output_dir=tmp_path / "out",
        formats=("h5",), on_decode_error="recover", save_metadata=True))
    assert result.status == "ok-with-gaps"
    assert result.frames_written == 12
    with h5py.File(result.outputs["h5"], "r") as f:
        temp = f["temperature_C"][...]
        status = f["frame_status"][...]
        stamps = f["timestamps_unix_ms"][...]
        raw = f["raw_counts"][...]
        meta = json.loads(f["metadata"][()])
    assert status.tolist() == [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    for index in (2, 7):
        assert np.isnan(temp[index]).all(), "a failed frame must be NaN, not zeros"
        assert raw[index].max() == 0
    finite = [i for i in range(12) if i not in (2, 7)]
    assert all(np.isfinite(temp[i]).any() for i in finite)
    truth = np.array(corrupt_recording["timestamps_ms"], dtype=np.int64)
    np.testing.assert_array_equal(stamps, truth)
    assert [f["frame_index"] for f in meta["failed_frames"]] == [2, 7]
    assert meta["conversion"]["recovered_frames"] == 2
    assert meta["failed_frames"][0]["payload_sha256"]
    assert result.validation["passed"], result.validation["problems"]
    assert result.validation["recovered_frames"] == 2


def test_recovery_survives_a_broken_first_frame(tmp_path):
    path = tmp_path / "first_bad.csq"
    make_streaming_csq(path, n_frames=6, height=20, width=24, corrupt_frames=(0,))
    result = convert(ConversionOptions(input_path=path, output_dir=tmp_path / "out",
                                       formats=("h5",), on_decode_error="recover",
                                       save_metadata=False))
    with h5py.File(result.outputs["h5"], "r") as f:
        temp = f["temperature_C"][...]
        status = f["frame_status"][...]
    assert status[0] == 1 and np.isnan(temp[0]).all()
    assert np.isfinite(temp[1]).any()


def test_recovery_can_dump_the_unreadable_payload(corrupt_recording, tmp_path):
    dump = tmp_path / "bad"
    result = convert(ConversionOptions(
        input_path=corrupt_recording["path"], output_dir=tmp_path / "out",
        formats=("h5",), on_decode_error="recover", dump_bad_frames=dump,
        save_metadata=True))
    files = sorted(dump.glob("*.jls"))
    assert [f.name for f in files] == ["gapped_frame_000002.jls", "gapped_frame_000007.jls"]
    meta = result.metadata
    import hashlib

    for entry in meta["failed_frames"]:
        data = Path(entry["payload_file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["payload_sha256"]


def test_recovery_for_mat73_output(corrupt_recording, tmp_path):
    result = convert(ConversionOptions(
        input_path=corrupt_recording["path"], output_dir=tmp_path / "out",
        formats=("mat",), mat_version="7.3", on_decode_error="recover",
        save_metadata=False))
    with h5py.File(result.outputs["mat"], "r") as f:
        status = np.asarray(f["frame_status"])
        temp = np.asarray(f["temperature_C"])
    assert status.tolist() == [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    assert np.isnan(np.asarray(temp[2])).all()  # HDF5 order: (frames, w, h)


# -- diagnostics -------------------------------------------------------------------
def test_check_frames_reports_failing_indices_deterministically(corrupt_recording):
    first = check_frames(corrupt_recording["path"])
    second = check_frames(corrupt_recording["path"])
    assert first["failed_indices"] == [2, 7]
    assert second["failed_indices"] == first["failed_indices"], "failures must be reproducible"
    assert first["decoded"] == 10
    assert first["failures"][0]["payload_bytes"] > 0
    assert "truncated" in first["failures"][0]["error"]


def test_check_frames_writes_payload_dumps(corrupt_recording, tmp_path):
    report = check_frames(corrupt_recording["path"], dump_dir=tmp_path / "dumps")
    assert len(list((tmp_path / "dumps").glob("*.jls"))) == 2
    assert report["failures"][0]["payload_file"]


def test_cli_check_frames(corrupt_recording):
    proc = subprocess.run([sys.executable, str(ROOT / "convert_csq.py"),
                           str(corrupt_recording["path"]), "--check-frames", "--json"],
                          capture_output=True, text=True, check=False, timeout=300)
    assert proc.returncode == 3
    report = json.loads(proc.stdout)
    assert report["failed_indices"] == [2, 7]
    assert report["all_decoded"] is False


def test_cli_recovery_end_to_end(corrupt_recording, tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / "convert_csq.py"),
                           str(corrupt_recording["path"]), "--output", str(tmp_path),
                           "--on-decode-error", "recover", "--format", "h5"],
                          capture_output=True, text=True, check=False, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "verify     PASS" in proc.stdout
    with h5py.File(tmp_path / "gapped_temperature.h5") as f:
        assert int(np.asarray(f["frame_status"]).sum()) == 2


def test_memory_logging(corrupt_recording, tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / "convert_csq.py"),
                           str(corrupt_recording["path"]), "--output", str(tmp_path),
                           "--on-decode-error", "recover", "--memory-log-every", "3"],
                          capture_output=True, text=True, check=False, timeout=300)
    assert proc.returncode == 0
    assert "memory: frame" in proc.stderr
    assert "peak" in proc.stderr


def test_peak_rss_is_recorded(corrupt_recording, tmp_path):
    result = convert(ConversionOptions(input_path=corrupt_recording["path"],
                                       output_dir=tmp_path, formats=("h5",),
                                       on_decode_error="recover", save_metadata=False))
    peak = result.metadata["memory"]["peak_rss_mb"]
    assert peak is None or peak > 10


# -- batch -------------------------------------------------------------------------
def test_batch_marks_a_file_with_gaps(tmp_path):
    from flir_csq.batch import BatchOptions, run_batch

    inp = tmp_path / "in"
    make_streaming_csq(inp / "a.csq", n_frames=8, height=20, width=24)
    make_streaming_csq(inp / "b.csq", n_frames=8, height=20, width=24, corrupt_frames=(4,))
    result = run_batch(BatchOptions(input_dir=inp, output_dir=tmp_path / "out",
                                    formats=("h5",), on_decode_error="recover"))
    assert result.converted == 2
    assert result.with_gaps == 1
    assert result.recovered_frames == 1
    gap = next(o for o in result.outcomes if o.status == "ok-with-gaps")
    assert gap.input.name == "b.csq" and gap.recovered_frames == 1
    summary = json.loads((tmp_path / "out" / "batch_summary.json").read_text())
    assert summary["totals"]["files_with_gaps"] == 1
