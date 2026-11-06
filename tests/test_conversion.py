"""End-to-end conversion tests: ground truth, ordering, timestamps, failures, reproducibility."""

from __future__ import annotations

import json

import numpy as np
import pytest

from flir_csq.converter import ConversionOptions, convert, inspect_recording
from flir_csq.errors import ConversionCancelled, CsqError, MissingCalibration
from flir_csq.synthetic import (
    SyntheticCamera,
    make_field,
    make_synthetic_csq,
)
from flir_csq.verify import verify_hdf5

h5py = pytest.importorskip("h5py")


def _convert(recording, out_dir, **kwargs):
    options = ConversionOptions(input_path=recording["path"], output_dir=out_dir,
                                formats=kwargs.pop("formats", ("h5",)), **kwargs)
    return convert(options)


def test_ground_truth_temperatures_are_recovered(jpegls_recording, out_dir):
    result = _convert(jpegls_recording, out_dir, save_metadata=True)
    with h5py.File(result.outputs["h5"], "r") as f:
        cube = f["temperature_C"][...]
        raw = f["raw_counts"][...]
    truth = jpegls_recording["temperatures_c"].astype(np.float32)
    assert cube.shape == truth.shape
    error = np.abs(cube - truth)
    assert float(np.nanmax(error)) < 0.02, f"max |T - truth| = {np.nanmax(error):.4f} degC"
    np.testing.assert_array_equal(raw, jpegls_recording["counts"])
    assert result.frames_written == 8
    assert result.validation["passed"]


def test_emissive_recording_uses_object_correction(emissive_recording, out_dir):
    result = _convert(emissive_recording, out_dir)
    assert result.mode_effective == "object"
    with h5py.File(result.outputs["h5"], "r") as f:
        cube = f["temperature_C"][...]
        meta = json.loads(f["metadata"][()])
    truth = emissive_recording["temperatures_c"].astype(np.float32)
    assert float(np.nanmax(np.abs(cube - truth))) < 0.02
    assert meta["conversion"]["corrections_applied"] is True
    assert meta["conversion"]["parameters"]["emissivity"] == pytest.approx(0.92)


def test_timestamps_preserve_camera_clock_units_and_are_exact(jpegls_recording, out_dir):
    result = _convert(jpegls_recording, out_dir)
    with h5py.File(result.outputs["h5"], "r") as f:
        seconds = f["timestamps"][...]
        millis = f["timestamps_unix_ms"][...]
        units = f["timestamps"].attrs["unit"]
    truth = np.array(jpegls_recording["timestamps_ms"], dtype=np.int64)
    np.testing.assert_array_equal(millis, truth)
    np.testing.assert_allclose(seconds, (truth - truth[0]) / 1000.0, atol=1e-9)
    assert units == "s"
    # integer-millisecond stamps at 33 ms give 30.303 Hz, which is the honest measurement
    assert result.metadata["timing"]["measured_frame_rate_hz"] == pytest.approx(30.3, rel=1e-3)
    assert result.metadata["timing"]["timestamps_estimated"] is False


def test_estimates_are_labelled_when_the_file_has_no_clock(tmp_path, out_dir):
    camera = SyntheticCamera(width=16, height=12, with_timestamps=False, frame_rate=25)
    path = tmp_path / "nots.csq"
    make_synthetic_csq(path, n_frames=4, height=12, width=16, compression="raw",
                       camera=camera, temperatures=make_field(4, 12, 16))
    result = convert(ConversionOptions(input_path=path, output_dir=out_dir, formats=("h5",)))
    timing = result.metadata["timing"]
    assert timing["timestamps_estimated"] is True
    assert "estimated" in timing["timestamp_source"]
    with h5py.File(result.outputs["h5"], "r") as f:
        seconds = f["timestamps"][...]
    np.testing.assert_allclose(seconds, np.arange(4) / 25.0, atol=1e-9)


def test_frame_subset_and_stride_keep_order(jpegls_recording, out_dir):
    result = _convert(jpegls_recording, out_dir, start=1, stop=7, stride=2)
    assert result.metadata["recording"]["converted_frame_indices"] == [1, 3, 5]
    with h5py.File(result.outputs["h5"], "r") as f:
        cube = f["temperature_C"][...]
        raw = f["raw_counts"][...]
    np.testing.assert_array_equal(raw, jpegls_recording["counts"][[1, 3, 5]])
    truth = jpegls_recording["temperatures_c"][[1, 3, 5]]
    np.testing.assert_allclose(cube, truth, atol=0.02)


def test_conversion_is_reproducible_bit_for_bit(jpegls_recording, tmp_path):
    a = _convert(jpegls_recording, tmp_path / "a")
    b = _convert(jpegls_recording, tmp_path / "b")
    with h5py.File(a.outputs["h5"], "r") as fa, h5py.File(b.outputs["h5"], "r") as fb:
        np.testing.assert_array_equal(fa["temperature_C"][...], fb["temperature_C"][...])
        np.testing.assert_array_equal(fa["timestamps_unix_ms"][...], fb["timestamps_unix_ms"][...])
        assert fa["temperature_C"].attrs["unit"] == fb["temperature_C"].attrs["unit"]


def test_incomplete_recording_converts_complete_frames_only(small_recording, out_dir):
    path = small_recording["path"]
    data = path.read_bytes()
    path.write_bytes(data[:-400])
    result = _convert(small_recording, out_dir)
    assert result.frames_written == 3
    assert any("incomplete" in w for w in result.warnings)
    with h5py.File(result.outputs["h5"], "r") as f:
        assert f["temperature_C"].shape[0] == 3


def test_missing_calibration_is_refused_and_raw_is_preserved(uncalibrated_recording, out_dir):
    with pytest.raises(MissingCalibration) as exc:
        _convert(uncalibrated_recording, out_dir, formats=("h5",))
    assert "planck" in str(exc.value).lower()
    assert not list(out_dir.glob("*_temperature.h5")), "no half-written cube may survive"

    result = _convert(uncalibrated_recording, out_dir, raw_only=True)
    assert result.status == "raw-only"
    with h5py.File(result.outputs["h5"], "r") as f:
        assert "temperature_C" not in f
        np.testing.assert_array_equal(f["raw_counts"][...], uncalibrated_recording["counts"])
        meta = json.loads(f["metadata"][()])
        assert meta["conversion"]["mode_effective"] == "raw-only"


def test_cancel_stops_and_leaves_no_output(jpegls_recording, out_dir):
    state = {"checked": 0}

    def cancel() -> bool:
        state["checked"] += 1
        return state["checked"] > 3  # cancel during frame 4

    with pytest.raises(ConversionCancelled):
        convert(ConversionOptions(input_path=jpegls_recording["path"], output_dir=out_dir,
                                  formats=("h5",), should_cancel=cancel))
    assert not list(out_dir.glob("*.h5"))
    assert not list(out_dir.glob("*.partial"))


def test_existing_output_requires_overwrite(jpegls_recording, out_dir):
    _convert(jpegls_recording, out_dir)
    with pytest.raises(CsqError):
        _convert(jpegls_recording, out_dir)
    _convert(jpegls_recording, out_dir, overwrite=True)


def test_all_formats_together(jpegls_recording, out_dir):
    result = _convert(jpegls_recording, out_dir, formats=("h5", "npy", "csv", "mat"))
    assert set(result.outputs) == {"h5", "npy", "csv", "mat", "json"}
    assert result.outputs["h5"].is_file()
    assert result.outputs["npy"].is_file()
    assert result.outputs["csv"].is_dir()
    assert result.outputs["mat"].is_file()
    assert result.outputs["json"].is_file()
    assert len(list(result.outputs["csv"].glob("frame_*.csv"))) == 8


def test_verification_detects_a_corrupted_cube(jpegls_recording, out_dir):
    result = _convert(jpegls_recording, out_dir)
    path = result.outputs["h5"]
    with h5py.File(path, "a") as f:
        f["temperature_C"][0, 0, 0] += 5.0
    report = verify_hdf5(path)
    assert report["passed"] is False
    assert any("reproducible" in p for p in report["problems"])


def test_inspection_reports_what_the_file_holds(jpegls_recording):
    info = inspect_recording(jpegls_recording["path"])
    assert info["frames"]["complete"] == 8
    assert info["dimensions"] == {"height": 48, "width": 64, "pixels": 48 * 64}
    assert info["camera"]["camera_model"] == "FLIR T540"
    assert info["radiometric"] is True
    assert info["temperature_conversion"] == "supported"
    assert info["jpeg_ls_supported"] is True
    assert info["timing"]["measured_frame_rate_hz"] == pytest.approx(30.3, rel=1e-2)


def test_inspection_flags_missing_calibration(uncalibrated_recording):
    info = inspect_recording(uncalibrated_recording["path"])
    assert info["temperature_conversion"] == "unsupported"
    assert set(info["missing_for_temperature"]) == {"planck_r1", "planck_b", "planck_f",
                                                    "planck_o", "planck_r2"}


def test_measurement_override_changes_the_result_and_is_recorded(emissive_recording, out_dir):
    base = _convert(emissive_recording, out_dir / "base")
    forced = _convert(emissive_recording, out_dir / "forced", emissivity=0.5)
    with h5py.File(base.outputs["h5"], "r") as f:
        t_base = f["temperature_C"][0, 0, 0]
    with h5py.File(forced.outputs["h5"], "r") as f:
        t_forced = f["temperature_C"][0, 0, 0]
        meta = json.loads(f["metadata"][()])
    assert abs(float(t_base) - float(t_forced)) > 0.5
    assert meta["conversion"]["parameters"]["emissivity"] == pytest.approx(0.5)
    assert any("emissivity overridden" in str(w) for w in meta["warnings"])


def test_metadata_contains_the_scientific_essentials(jpegls_recording, out_dir):
    result = _convert(jpegls_recording, out_dir)
    meta = result.metadata
    assert meta["camera"]["camera_model"] == "FLIR T540"
    assert meta["camera"]["serial_number"] == "SYNTH0001"
    assert meta["geometry"]["shape"] == [8, 48, 64]
    assert meta["conversion"]["units"] == "degC"
    assert meta["conversion"]["coefficients"]["planck_r1"] == pytest.approx(11895.470703125)
    assert meta["recording"]["fingerprint"]["size_bytes"] > 0
    assert meta["converter"]["version"]
    assert "per_frame" in meta["statistics"]


def test_stride_larger_than_recording_is_rejected(jpegls_recording, out_dir):
    with pytest.raises(CsqError):
        _convert(jpegls_recording, out_dir, start=50)
