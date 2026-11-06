"""Container and payload decoding tests."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from flir_csq.codecs import classify_payload, jpeg_profile
from flir_csq.container import CsqReader, scan_csq
from flir_csq.errors import NotACsqFile
from flir_csq.fff import HEADER_SIZE, RECORD_SIZE
from flir_csq.metadata import parse_camera_info
from flir_csq.synthetic import SyntheticCamera, build_frame


def test_scan_finds_every_frame_and_contiguous_offsets(jpegls_recording):
    index = scan_csq(jpegls_recording["path"])
    assert index.n_frames == 8
    assert index.complete_frames == 8
    assert index.creator == "CAMCTRL"
    assert index.format_version == 101
    assert index.leading_bytes == 0
    assert index.trailing_bytes == 0
    offsets = [f.offset for f in index.frames]
    assert offsets == sorted(offsets)
    for previous, current in zip(index.frames, index.frames[1:]):
        assert previous.end == current.offset, "frames must tile the file without gaps"
    assert index.frames[-1].end == index.file_size


def test_jpegls_payload_roundtrips_exactly(jpegls_recording):
    truth = jpegls_recording["counts"]
    with CsqReader(scan_csq(jpegls_recording["path"])) as reader:
        for i in range(len(reader)):
            frame = reader.read_frame(i)
            assert frame.codec.startswith("imagecodecs")
            assert frame.raw.dtype == np.uint16
            assert frame.raw.shape == truth.shape[1:]
            np.testing.assert_array_equal(frame.raw, truth[i])


def test_uncompressed_payload_decodes_to_identical_counts(emissive_recording):
    truth = emissive_recording["counts"]
    with CsqReader(scan_csq(emissive_recording["path"])) as reader:
        for i in range(len(reader)):
            frame = reader.read_frame(i)
            assert "raw" in frame.codec
            np.testing.assert_array_equal(frame.raw, truth[i])


def test_frame_order_is_preserved(tmp_path):
    """Frames are written with increasing brightness; order must survive the pipeline."""
    frames = np.stack([np.full((8, 10), 14000 + 37 * i, dtype=np.uint16) for i in range(6)])
    path = tmp_path / "ramp.csq"
    camera = SyntheticCamera(width=10, height=8)
    blocks = [build_frame(frames[i], camera, timestamp_ms=1_700_000_000_000 + i * 33,
                          compression="raw") for i in range(6)]
    path.write_bytes(b"".join(blocks))
    with CsqReader(scan_csq(path)) as reader:
        means = [float(reader.read_frame(i).raw.mean()) for i in range(6)]
    assert means == sorted(means)
    assert all(abs(m - 14000 - 37 * i) < 0.01 for i, m in enumerate(means))


def test_camera_record_matches_what_was_written(emissive_recording):
    camera = emissive_recording["camera"]
    with CsqReader(scan_csq(emissive_recording["path"])) as reader:
        info = reader.camera_info(0)
    assert info.camera_model == "FLIR T540"
    assert info.serial_number == "SYNTH0001"
    assert info.emissivity == pytest.approx(0.92, abs=1e-6)
    assert info.object_distance_m == pytest.approx(1.0, abs=1e-6)
    assert info.relative_humidity == pytest.approx(0.6, abs=1e-3)
    assert info.planck_r1 == pytest.approx(camera.calibration["planck_r1"], rel=1e-6)
    assert info.planck_o == camera.calibration["planck_o"]
    assert info.has_planck
    assert info.raw_value_range_min == 5589
    assert info.raw_value_range_max == 53132
    assert info.frame_rate_declared == 30
    assert info.field_of_view_deg == pytest.approx(42.0, abs=1e-3)


def test_timestamps_are_read_per_frame_with_millisecond_resolution(jpegls_recording):
    truth = jpegls_recording["timestamps_ms"]
    with CsqReader(scan_csq(jpegls_recording["path"])) as reader:
        stamps = [reader.camera_info(i).timestamp_unix_ms for i in range(len(reader))]
    assert stamps == truth
    diffs = np.diff(stamps)
    assert set(diffs.tolist()) == {33}


def test_truncated_last_frame_is_marked_not_silently_dropped(small_recording):
    path = small_recording["path"]
    data = path.read_bytes()
    path.write_bytes(data[:-500])
    index = scan_csq(path)
    assert index.n_frames == 4
    assert index.complete_frames == 3
    assert index.frames[-1].truncated
    assert any("incomplete" in w for w in index.warnings)


def test_garbage_between_frames_is_skipped_with_a_warning(jpegls_recording, tmp_path):
    data = jpegls_recording["path"].read_bytes()
    first_frame_size = scan_csq(jpegls_recording["path"]).frames[0].size
    poisoned = data[:first_frame_size] + b"\x00" * 64 + data[first_frame_size:]
    path = tmp_path / "poisoned.csq"
    path.write_bytes(poisoned)
    index = scan_csq(path)
    assert index.n_frames == 8
    assert any("unparsable" in w for w in index.warnings)


def test_non_csq_file_is_rejected(tmp_path):
    path = tmp_path / "not_a_recording.csq"
    path.write_bytes(b"this is not a FLIR container" * 100)
    with pytest.raises(NotACsqFile):
        scan_csq(path)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.csq"
    path.write_bytes(b"")
    with pytest.raises(NotACsqFile):
        scan_csq(path)


def test_camera_info_of_truncated_record_does_not_raise():
    info = parse_camera_info(b"\x02\x00\x10\x00")  # 4 bytes only
    assert info.camera_model is None
    assert info.planck_r1 is None
    assert not info.has_planck
    assert info.warnings


def test_geometry_change_is_detected(tmp_path):
    """A file whose second frame has different dimensions must not be averaged into one cube."""
    camera = SyntheticCamera(width=10, height=8)
    a = build_frame(np.full((8, 10), 14000, np.uint16), camera, timestamp_ms=1_700_000_000_000,
                    compression="raw")
    camera2 = SyntheticCamera(width=12, height=8)
    b = build_frame(np.full((8, 12), 14000, np.uint16), camera2, timestamp_ms=1_700_000_000_100,
                    compression="raw")
    path = tmp_path / "mixed.csq"
    path.write_bytes(a + b)
    from flir_csq.converter import ConversionOptions, convert
    from flir_csq.errors import CsqError

    with pytest.raises(CsqError):
        convert(ConversionOptions(input_path=path, output_dir=tmp_path,
                                  formats=("h5",), save_metadata=False))


def test_magic_and_record_offsets_are_consistent(jpegls_recording):
    data = jpegls_recording["path"].read_bytes()
    header = struct.unpack_from("<4s16sIIIIH7HIII", data, 0)
    assert header[0] == b"FFF\x00"
    assert header[3] == HEADER_SIZE == 64
    assert header[4] >= 1
    assert RECORD_SIZE == 32
    index = scan_csq(jpegls_recording["path"])
    assert index.record_types[0x01] == 8
    assert index.record_types[0x20] == 8


def test_jpeg_profile_detects_jpegls(jpegls_recording, tmp_path):
    import imagecodecs

    payload = imagecodecs.jpegls_encode(np.zeros((4, 4), dtype=np.uint16))
    assert classify_payload(payload) == "jpegls"
    assert jpeg_profile(payload) == "jpegls"
    assert classify_payload(b"\x89PNG\r\n\x1a\n") == "png"
    assert classify_payload(np.zeros((3, 3), np.uint16).tobytes()) == "raw16"


def test_two_codecs_agree_on_the_same_payload(jpegls_recording):
    """imagecodecs/CharLS and pylibjpeg-libjpeg must return identical counts."""
    libjpeg = pytest.importorskip("libjpeg")
    import imagecodecs

    with CsqReader(scan_csq(jpegls_recording["path"])) as reader:
        frame = reader.read_frame(0)
        payload = _payload_of(reader, 0)
    reference = imagecodecs.jpegls_decode(payload)
    np.testing.assert_array_equal(np.asarray(libjpeg.decode(payload)), reference)
    np.testing.assert_array_equal(frame.raw, reference)


def _payload_of(reader, i):
    from flir_csq.fff import RAW_DATA, find_record, parse_header, parse_records

    data = reader.frame_bytes(i)
    rec = find_record(parse_records(data, 0, parse_header(data)), RAW_DATA)
    assert rec is not None
    return data[rec.offset:rec.offset + rec.length][0x20:]


def test_read_frame_out_of_range_raises(jpegls_recording):
    with CsqReader(scan_csq(jpegls_recording["path"])) as reader, pytest.raises(IndexError):
        reader.read_frame(99)
