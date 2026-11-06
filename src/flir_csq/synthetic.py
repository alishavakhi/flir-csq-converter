"""Synthetic CSQ recordings for tests and the ``--demo`` mode.

These files are built with exactly the container layout documented in
:mod:`flir_csq.fff` and the field offsets of :mod:`flir_csq.metadata`, so they
exercise the real parser - not a mock of it. They are still **synthetic**: a
passing test on them proves the implementation is self-consistent and that the
radiometric inversion round-trips the forward model, and proves nothing about
any particular camera firmware. Real-camera validation is documented in
``docs/validation.md``.

The calibration constants used by default are copied from a real public FLIR
T1020 ``.csq`` recording (R1 = 11895.470703125, B = 1328.9, F = 1, O = -3869,
R2 = 0.013583816587924957) purely so that count values and temperature ranges
are physically realistic. They are never used when converting a real file: a
conversion always reads the constants from that file.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .fff import HEADER, HEADER_SIZE, RECORD, RECORD_SIZE
from .radiometry import atmospheric_transmission

#: Constants of a real FLIR T1020 recording - synthetic data only.
DEMO_CALIBRATION = {
    "planck_r1": 11895.470703125,
    "planck_b": 1328.9000244140625,
    "planck_f": 1.0,
    "planck_o": -3869,
    "planck_r2": 0.013583816587924957,
}

DEMO_EPOCH_MS = 1_700_000_000_000  # 2023-11-14T22:13:20Z


@dataclass
class SyntheticCamera:
    """Everything the generator writes into the 0x20 record."""

    model: str = "FLIR T540"
    serial: str = "SYNTH0001"
    width: int = 464
    height: int = 348
    lens_model: str = "FOL15"
    field_of_view_deg: float = 42.0
    emissivity: float = 1.0
    object_distance_m: float = 1.0
    reflected_temperature_c: float = 20.0
    atmospheric_temperature_c: float = 20.0
    relative_humidity: float = 0.5
    window_transmission: float = 1.0
    window_temperature_c: float = 20.0
    frame_rate: int = 30
    raw_range_min: int = 5589
    raw_range_max: int = 53132
    calibration: dict[str, float] = field(default_factory=lambda: dict(DEMO_CALIBRATION))
    with_planck: bool = True
    with_timestamps: bool = True


def temperatures_to_counts(temperatures_c: np.ndarray, camera: SyntheticCamera) -> np.ndarray:
    """Forward model: object temperature -> sensor counts, as the camera would store them."""
    c = camera.calibration
    r1, b, f, o, r2 = c["planck_r1"], c["planck_b"], c["planck_f"], c["planck_o"], c["planck_r2"]

    def signal(t_c: float | np.ndarray) -> np.ndarray:
        return r1 / (r2 * (np.exp(b / (np.asarray(t_c, dtype=np.float64) + 273.15)) - f))

    eps = camera.emissivity
    tau_win = camera.window_transmission
    # Same forward model the converter inverts: object, reflected, atmosphere and window.
    tau = atmospheric_transmission(camera.object_distance_m, camera.atmospheric_temperature_c,
                                   camera.relative_humidity)
    c_meas = tau_win * (eps * tau * signal(temperatures_c)
                        + (1 - eps) * tau * signal(camera.reflected_temperature_c)
                        + (1 - tau) * signal(camera.atmospheric_temperature_c)) \
        + (1 - tau_win) * signal(camera.window_temperature_c)
    counts = np.rint(c_meas - o)
    return np.clip(counts, 0, 65535).astype(np.uint16)


def make_field(n_frames: int, height: int, width: int, *,
               base_c: float = 25.0, amplitude_c: float = 12.0,
               hot_spot: bool = True) -> np.ndarray:
    """A drifting thermal scene with a warm spot - smooth, so JPEG-LS behaves realistically."""
    y, x = np.mgrid[0:height, 0:width]
    norm_x = x / max(width - 1, 1)
    norm_y = y / max(height - 1, 1)
    out = np.empty((n_frames, height, width), dtype=np.float64)
    for i in range(n_frames):
        phase = 2 * math.pi * i / max(n_frames, 1)
        frame = base_c + amplitude_c * (0.35 * norm_x + 0.25 * norm_y) + 1.5 * math.sin(phase)
        if hot_spot:
            cx, cy = 0.62 * width, 0.45 * height
            sigma = 0.09 * min(height, width)
            frame = frame + 18.0 * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)))
        out[i] = frame
    return out


def build_camera_record(camera: SyntheticCamera, *, timestamp_ms: int | None,
                        tz_minutes: int = 0, size: int = 2476) -> bytes:
    """Serialise a 0x20 CameraInfo record with the documented offsets."""
    blob = bytearray(size)

    def put(fmt: str, offset: int, value: Any) -> None:
        struct.pack_into(fmt, blob, offset, value)

    def put_str(text: str, offset: int, length: int) -> None:
        raw = text.encode("ascii")[: length - 1]
        blob[offset:offset + len(raw)] = raw

    put("<H", 0x00, 0x0002)  # little-endian sentinel
    put("<H", 0x02, camera.width)
    put("<H", 0x04, camera.height)
    put("<f", 0x20, camera.emissivity)
    put("<f", 0x24, camera.object_distance_m)
    put("<f", 0x28, camera.reflected_temperature_c + 273.15)
    put("<f", 0x2C, camera.atmospheric_temperature_c + 273.15)
    put("<f", 0x30, camera.window_temperature_c + 273.15)
    put("<f", 0x34, camera.window_transmission)
    put("<f", 0x3C, camera.relative_humidity if camera.relative_humidity <= 1 else
        camera.relative_humidity / 100.0)

    if camera.with_planck:
        put("<f", 0x58, camera.calibration["planck_r1"])
        put("<f", 0x5C, camera.calibration["planck_b"])
        put("<f", 0x60, camera.calibration["planck_f"])
        put("<i", 0x308, int(camera.calibration["planck_o"]))
        put("<f", 0x30C, camera.calibration["planck_r2"])

    put("<f", 0x70, 0.006569)
    put("<f", 0x74, 0.012620)
    put("<f", 0x78, -0.002276)
    put("<f", 0x7C, -0.006670)
    put("<f", 0x80, 1.900000)
    put("<f", 0x90, 423.15)  # 150 degC
    put("<f", 0x94, 233.15)  # -40 degC
    put_str(camera.model, 0xD4, 32)
    put_str("SYN-PART", 0xF4, 16)
    put_str(camera.serial, 0x104, 16)
    put_str("synthetic-1.0", 0x114, 16)
    put_str(camera.lens_model, 0x170, 32)
    put("<f", 0x1B4, camera.field_of_view_deg)
    put("<H", 0x310, camera.raw_range_min)
    put("<H", 0x312, camera.raw_range_max)
    put("<H", 0x338, 14000)
    put("<H", 0x33C, 2500)
    if camera.with_timestamps and timestamp_ms is not None:
        seconds = timestamp_ms // 1000
        millis = timestamp_ms % 1000
        put("<I", 0x384, seconds)
        put("<I", 0x388, millis)
        put("<h", 0x38C, tz_minutes)
    put("<H", 0x464, camera.frame_rate)
    return bytes(blob)


def _encode_payload(counts: np.ndarray, compression: str) -> tuple[bytes, int]:
    """Return (payload, subtype) for the configured payload codec."""
    compression = compression.lower()
    if compression in ("raw", "raw16", "none"):
        return counts.astype("<u2").tobytes(), 2
    if compression == "png":
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(counts.astype(np.uint16)).save(buf, format="PNG")
        return buf.getvalue(), 3
    if compression in ("jpegls", "jpeg-ls"):
        import imagecodecs

        return imagecodecs.jpegls_encode(counts.astype(np.uint16)), 4
    raise ValueError(f"unknown synthetic compression {compression!r}")


def build_frame(counts: np.ndarray, camera: SyntheticCamera, *, timestamp_ms: int | None,
                compression: str = "jpegls", tz_minutes: int = 0,
                payload_override: bytes | None = None) -> bytes:
    """Serialise one FFF block. ``payload_override`` lets tests inject a broken frame."""
    height, width = counts.shape
    if payload_override is not None:
        payload, subtype = payload_override, 4
    else:
        payload, subtype = _encode_payload(counts, compression)
    raw_record_size = 0x20 + len(payload)
    camera_record = build_camera_record(camera, timestamp_ms=timestamp_ms, tz_minutes=tz_minutes)
    n_records = 2
    dir_offset = HEADER_SIZE
    data_offset = dir_offset + n_records * RECORD_SIZE
    raw_offset = data_offset
    cam_offset = raw_offset + raw_record_size
    total = cam_offset + len(camera_record)

    header = HEADER.pack(b"FFF\x00", b"CAMCTRL".ljust(16, b"\x00"), 101, dir_offset, n_records,
                         2, 0, *(0,) * 7, 0, 0, 0)
    raw_rec = RECORD.pack(0x01, subtype, 0x64, 1, raw_offset, raw_record_size, 0, 0, 0)
    cam_rec = RECORD.pack(0x20, 0, 0x74, 2, cam_offset, len(camera_record), 0, 0, 0)

    raw_header = bytearray(0x20)
    struct.pack_into("<H", raw_header, 0x00, 0x0002)  # little-endian
    struct.pack_into("<H", raw_header, 0x02, width)
    struct.pack_into("<H", raw_header, 0x04, height)
    struct.pack_into("<H", raw_header, 0x06, width - 1)
    struct.pack_into("<H", raw_header, 0x08, height - 1)

    block = b"".join([header, raw_rec, cam_rec, bytes(raw_header), payload, camera_record])
    if len(block) != total:  # pragma: no cover - structural self-check
        raise AssertionError(f"synthetic frame length {len(block)} != computed {total}")
    return block


def make_synthetic_csq(
    path: str | Path,
    *,
    n_frames: int = 5,
    height: int = 32,
    width: int = 24,
    compression: str = "jpegls",
    camera: SyntheticCamera | None = None,
    temperatures: np.ndarray | None = None,
    frame_interval_s: float | None = None,
    epoch_ms: int = DEMO_EPOCH_MS,
    tz_minutes: int = 0,
) -> dict[str, Any]:
    """Write a synthetic ``.csq`` and return the ground truth used to build it."""
    p = Path(path)
    if camera is None:
        camera = SyntheticCamera(width=width, height=height)
    camera.width, camera.height = width, height
    if temperatures is None:
        temperatures = make_field(n_frames, height, width)
    temperatures = np.asarray(temperatures, dtype=np.float64)
    if temperatures.ndim == 2:
        temperatures = temperatures[None, ...]
    n_frames = temperatures.shape[0]
    counts = temperatures_to_counts(temperatures, camera)
    interval_ms = round((frame_interval_s if frame_interval_s is not None
                         else 1.0 / max(camera.frame_rate, 1)) * 1000)

    blocks = []
    timestamps: list[int | None] = []
    for i in range(n_frames):
        ts = epoch_ms + i * interval_ms if camera.with_timestamps else None
        timestamps.append(ts)
        blocks.append(build_frame(counts[i], camera, timestamp_ms=ts,
                                  compression=compression, tz_minutes=tz_minutes))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"".join(blocks))
    return {
        "path": p,
        "temperatures_c": temperatures,
        "counts": counts,
        "timestamps_ms": timestamps,
        "camera": camera,
        "compression": compression,
        "n_frames": n_frames,
        "interval_ms": interval_ms,
    }


def write_demo(path: str | Path, *, n_frames: int = 12, width: int = 464,
               height: int = 348) -> dict[str, Any]:
    """The ``--demo`` recording: T540 geometry, JPEG-LS payload, 30 Hz timestamps."""
    return make_synthetic_csq(path, n_frames=n_frames, height=height, width=width,
                              compression="jpegls",
                              temperatures=make_field(n_frames, height, width))


def frame_field(index: int, height: int, width: int, *, base_c: float = 25.0,
                amplitude_c: float = 12.0, noise_c: float = 0.0,
                seed: int = 0) -> np.ndarray:
    """One frame of the synthetic scene, computed without materialising the recording."""
    y, x = np.mgrid[0:height, 0:width]
    norm_x = x / max(width - 1, 1)
    norm_y = y / max(height - 1, 1)
    phase = 2 * math.pi * index / 24.0
    frame = base_c + amplitude_c * (0.35 * norm_x + 0.25 * norm_y) + 1.5 * math.sin(phase)
    cx, cy = 0.62 * width, 0.45 * height
    sigma = 0.09 * min(height, width)
    frame = frame + 18.0 * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)))
    if noise_c:
        rng = np.random.default_rng(seed + index)
        frame = frame + rng.normal(0.0, noise_c, frame.shape)
    return frame


def make_streaming_csq(
    path: str | Path,
    *,
    n_frames: int,
    height: int = 348,
    width: int = 464,
    compression: str = "jpegls",
    camera: SyntheticCamera | None = None,
    frame_interval_s: float | None = None,
    epoch_ms: int = DEMO_EPOCH_MS,
    tz_minutes: int = 0,
    noise_c: float = 0.0,
    corrupt_frames: tuple[int, ...] = (),
    truncate_payload_by: int = 48,
) -> dict[str, Any]:
    """Write a long synthetic recording **one frame at a time** (constant memory).

    ``corrupt_frames`` truncates the JPEG-LS payload of those frames so decoding fails
    deterministically at a known index - the fixture for the recovery tests.
    """
    p = Path(path)
    if camera is None:
        camera = SyntheticCamera(width=width, height=height)
    camera.width, camera.height = width, height
    interval_ms = round((frame_interval_s if frame_interval_s is not None
                         else 1.0 / max(camera.frame_rate, 1)) * 1000)
    p.parent.mkdir(parents=True, exist_ok=True)
    bad = {int(i) for i in corrupt_frames}
    timestamps: list[int | None] = []
    with p.open("wb") as fh:
        for i in range(n_frames):
            field = frame_field(i, height, width, noise_c=noise_c, seed=0)
            counts = temperatures_to_counts(field, camera)
            payload_override = None
            if i in bad:
                import imagecodecs

                good = imagecodecs.jpegls_encode(counts)
                payload_override = good[:-truncate_payload_by] if len(good) > truncate_payload_by \
                    else good[:8]
            ts = epoch_ms + i * interval_ms if camera.with_timestamps else None
            timestamps.append(ts)
            fh.write(build_frame(counts, camera, timestamp_ms=ts, compression=compression,
                                 tz_minutes=tz_minutes, payload_override=payload_override))
    return {
        "path": p,
        "camera": camera,
        "n_frames": n_frames,
        "interval_ms": interval_ms,
        "timestamps_ms": timestamps,
        "corrupt_frames": tuple(sorted(bad)),
        "height": height,
        "width": width,
    }
