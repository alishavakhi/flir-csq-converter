"""Interpretation of the FFF CameraInfo record (type 0x20).

Offsets are the public ExifTool ``%Image::ExifTool::FLIR::CameraInfo`` table,
quoted field by field in :data:`CAMERA_FIELD_OFFSETS`. Everything here is
read *per frame* from the recording; nothing is hardcoded from another camera
and no default is silently substituted for a missing constant.

The record is a fixed-layout little-endian structure (endianness is declared by
the first uint16, which is 0x0002 when the record is little-endian).
"""

from __future__ import annotations

import datetime as _dt
import struct
from dataclasses import asdict, dataclass, field
from typing import Any

#: (attribute, offset, kind) - offsets from the start of the 0x20 record.
CAMERA_FIELD_OFFSETS: tuple[tuple[str, int, str], ...] = (
    ("raw_width", 0x02, "u16"),
    ("raw_height", 0x04, "u16"),
    ("emissivity", 0x20, "f32"),
    ("object_distance_m", 0x24, "f32"),
    ("reflected_apparent_temperature_k", 0x28, "f32"),
    ("atmospheric_temperature_k", 0x2C, "f32"),
    ("ir_window_temperature_k", 0x30, "f32"),
    ("ir_window_transmission", 0x34, "f32"),
    ("relative_humidity_raw", 0x3C, "f32"),
    ("planck_r1", 0x58, "f32"),
    ("planck_b", 0x5C, "f32"),
    ("planck_f", 0x60, "f32"),
    ("atmospheric_trans_alpha1", 0x70, "f32"),
    ("atmospheric_trans_alpha2", 0x74, "f32"),
    ("atmospheric_trans_beta1", 0x78, "f32"),
    ("atmospheric_trans_beta2", 0x7C, "f32"),
    ("atmospheric_trans_x", 0x80, "f32"),
    ("camera_range_max_k", 0x90, "f32"),
    ("camera_range_min_k", 0x94, "f32"),
    ("camera_max_clip_k", 0x98, "f32"),
    ("camera_min_clip_k", 0x9C, "f32"),
    ("camera_max_saturated_k", 0xA8, "f32"),
    ("camera_min_saturated_k", 0xAC, "f32"),
    ("field_of_view_deg", 0x1B4, "f32"),
    ("planck_o", 0x308, "i32"),
    ("planck_r2", 0x30C, "f32"),
    ("raw_value_range_min", 0x310, "u16"),
    ("raw_value_range_max", 0x312, "u16"),
    ("raw_value_median", 0x338, "u16"),
    ("raw_value_range", 0x33C, "u16"),
    ("focus_distance_m", 0x45C, "f32"),
    ("frame_rate_declared", 0x464, "u16"),
)

CAMERA_STRING_OFFSETS: tuple[tuple[str, int, int], ...] = (
    ("camera_model", 0xD4, 32),
    ("camera_part_number", 0xF4, 16),
    ("serial_number", 0x104, 16),
    ("software", 0x114, 16),
    ("lens_model", 0x170, 32),
    ("lens_part_number", 0x190, 16),
    ("lens_serial_number", 0x1A0, 16),
    ("filter_model", 0x1EC, 16),
    ("filter_part_number", 0x1FC, 32),
)

TIMESTAMP_OFFSET = 0x384  # uint32 unix seconds, uint32 ms in the low half, int16 tz minutes
MIN_RECORD_BYTES = 0x320  # everything up to RawValueRangeMin/Max


@dataclass
class CameraInfo:
    """Radiometric and measurement metadata carried by one frame."""

    # measurement parameters
    emissivity: float | None = None
    object_distance_m: float | None = None
    reflected_apparent_temperature_c: float | None = None
    atmospheric_temperature_c: float | None = None
    ir_window_temperature_c: float | None = None
    ir_window_transmission: float | None = None
    relative_humidity: float | None = None
    # Planck calibration
    planck_r1: float | None = None
    planck_b: float | None = None
    planck_f: float | None = None
    planck_o: float | None = None
    planck_r2: float | None = None
    # atmospheric transmission model
    atmospheric_trans_alpha1: float | None = None
    atmospheric_trans_alpha2: float | None = None
    atmospheric_trans_beta1: float | None = None
    atmospheric_trans_beta2: float | None = None
    atmospheric_trans_x: float | None = None
    # camera limits and identity
    camera_model: str | None = None
    serial_number: str | None = None
    camera_part_number: str | None = None
    software: str | None = None
    lens_model: str | None = None
    lens_part_number: str | None = None
    lens_serial_number: str | None = None
    filter_model: str | None = None
    filter_part_number: str | None = None
    field_of_view_deg: float | None = None
    camera_temp_range_min_c: float | None = None
    camera_temp_range_max_c: float | None = None
    raw_value_range_min: int | None = None
    raw_value_range_max: int | None = None
    raw_value_median: int | None = None
    raw_value_range: int | None = None
    # timing
    timestamp_unix_ms: int | None = None
    timestamp_tz_minutes: int | None = None
    frame_rate_declared: int | None = None
    # bookkeeping
    record_bytes: int = 0
    big_endian: bool = False
    warnings: list[str] = field(default_factory=list)

    # -- derived ---------------------------------------------------------------------
    @property
    def has_planck(self) -> bool:
        """True when the Planck inversion is defined for this record."""
        values = (self.planck_r1, self.planck_b, self.planck_f, self.planck_o, self.planck_r2)
        if any(v is None for v in values):
            return False
        return bool(self.planck_b) and bool(self.planck_r1) and bool(self.planck_r2)

    @property
    def timestamp_utc(self) -> _dt.datetime | None:
        if self.timestamp_unix_ms is None:
            return None
        return _dt.datetime.fromtimestamp(self.timestamp_unix_ms / 1000.0, tz=_dt.timezone.utc)

    @property
    def timestamp_local(self) -> _dt.datetime | None:
        ts = self.timestamp_utc
        if ts is None:
            return None
        tz = _dt.timezone(_dt.timedelta(minutes=self.timestamp_tz_minutes or 0))
        return ts.astimezone(tz)

    def planck_dict(self) -> dict[str, float] | None:
        if not self.has_planck:
            return None
        return {"r1": float(self.planck_r1), "b": float(self.planck_b), "f": float(self.planck_f),
                "o": float(self.planck_o), "r2": float(self.planck_r2)}

    def calibration_signature(self) -> tuple:
        """Values that must not change mid-recording for a single calibration to apply."""
        return (self.planck_r1, self.planck_b, self.planck_f, self.planck_o, self.planck_r2,
                self.raw_value_range_min, self.raw_value_range_max)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp_utc"] = self.timestamp_utc.isoformat() if self.timestamp_utc else None
        d["timestamp_local"] = self.timestamp_local.isoformat() if self.timestamp_local else None
        d["has_planck"] = self.has_planck
        return d


def _f32(blob: bytes, off: int, big: bool) -> float | None:
    if len(blob) < off + 4:
        return None
    return struct.unpack_from(">f" if big else "<f", blob, off)[0]


def _u16(blob: bytes, off: int, big: bool) -> int | None:
    if len(blob) < off + 2:
        return None
    return struct.unpack_from(">H" if big else "<H", blob, off)[0]


def _i32(blob: bytes, off: int, big: bool) -> int | None:
    if len(blob) < off + 4:
        return None
    return struct.unpack_from(">i" if big else "<i", blob, off)[0]


def _string(blob: bytes, off: int, size: int) -> str | None:
    if len(blob) < off + 1:
        return None
    chunk = blob[off:off + size].split(b"\x00", 1)[0]
    text = chunk.decode("ascii", errors="replace").strip()
    if not text or not text.isprintable():
        return None
    return text


def _kelvin_to_celsius(value: float | None) -> float | None:
    if value is None:
        return None
    if not (-50.0 <= value <= 1200.0):  # kelvin values outside this are not temperatures
        return None
    return round(value - 273.15, 4)


def _plausible(value: float | None, lo: float, hi: float) -> float | None:
    if value is None:
        return None
    if not (lo <= value <= hi):
        return None
    return float(value)


def parse_camera_info(blob: bytes) -> CameraInfo:
    """Parse a 0x20 CameraInfo record. Never raises on short/odd records.

    Absent or implausible fields stay ``None`` and a warning is recorded, so
    the conversion layer can refuse rather than invent a calibration.
    """
    info = CameraInfo(record_bytes=len(blob))
    if len(blob) < 4:
        info.warnings.append("camera-information record is empty")
        return info
    big = struct.unpack_from("<H", blob, 0)[0] != 0x0002
    info.big_endian = big
    if len(blob) < MIN_RECORD_BYTES:
        info.warnings.append(
            f"camera-information record is {len(blob)} bytes, shorter than the {MIN_RECORD_BYTES} "
            "bytes the documented layout needs; missing fields stay empty"
        )

    raw: dict[str, Any] = {}
    for name, off, kind in CAMERA_FIELD_OFFSETS:
        if kind == "f32":
            raw[name] = _f32(blob, off, big)
        elif kind == "u16":
            raw[name] = _u16(blob, off, big)
        else:
            raw[name] = _i32(blob, off, big)
    for name, off, size in CAMERA_STRING_OFFSETS:
        raw[name] = _string(blob, off, size)

    info.emissivity = _plausible(raw["emissivity"], 0.01, 1.0)
    info.object_distance_m = _plausible(raw["object_distance_m"], 0.0, 1.0e5)
    info.reflected_apparent_temperature_c = _kelvin_to_celsius(raw["reflected_apparent_temperature_k"])
    info.atmospheric_temperature_c = _kelvin_to_celsius(raw["atmospheric_temperature_k"])
    info.ir_window_temperature_c = _kelvin_to_celsius(raw["ir_window_temperature_k"])
    info.ir_window_transmission = _plausible(raw["ir_window_transmission"], 0.01, 1.0)
    rh = raw["relative_humidity_raw"]
    if rh is not None:
        # Have seen both 0-1 and 0-100 in the wild; ExifTool divides by 100 when > 2.
        info.relative_humidity = _plausible(rh / 100.0 if rh > 2 else rh, 0.0, 1.0)

    # A zero-filled calibration block means "no calibration", not "constant = 0":
    # none of R1, B, F, R2 is physically zero, and O alone is meaningless.
    info.planck_r1 = _plausible(raw["planck_r1"], 1.0, 1.0e6)
    info.planck_b = _plausible(raw["planck_b"], 1.0, 1.0e5)
    info.planck_f = _plausible(raw["planck_f"], 1.0e-6, 1.0e4)
    info.planck_o = raw["planck_o"] if raw["planck_o"] is not None and abs(raw["planck_o"]) < 1e6 else None
    info.planck_r2 = _plausible(raw["planck_r2"], 1.0e-9, 1.0e3)
    if info.planck_r1 is None:
        info.planck_b = info.planck_f = info.planck_o = info.planck_r2 = None

    info.atmospheric_trans_alpha1 = raw["atmospheric_trans_alpha1"]
    info.atmospheric_trans_alpha2 = raw["atmospheric_trans_alpha2"]
    info.atmospheric_trans_beta1 = raw["atmospheric_trans_beta1"]
    info.atmospheric_trans_beta2 = raw["atmospheric_trans_beta2"]
    info.atmospheric_trans_x = _plausible(raw["atmospheric_trans_x"], 0.0, 1.0)

    info.camera_range_max_k = raw["camera_range_max_k"]
    info.camera_range_min_k = raw["camera_range_min_k"]
    info.camera_temp_range_max_c = _kelvin_to_celsius(raw["camera_range_max_k"])
    info.camera_temp_range_min_c = _kelvin_to_celsius(raw["camera_range_min_k"])

    info.camera_model = raw["camera_model"]
    info.serial_number = raw["serial_number"]
    info.camera_part_number = raw["camera_part_number"]
    info.software = raw["software"]
    info.lens_model = raw["lens_model"]
    info.lens_part_number = raw["lens_part_number"]
    info.lens_serial_number = raw["lens_serial_number"]
    info.filter_model = raw["filter_model"]
    info.filter_part_number = raw["filter_part_number"]
    info.field_of_view_deg = _plausible(raw["field_of_view_deg"], 0.1, 179.9)

    for key in ("raw_value_range_min", "raw_value_range_max", "raw_value_median", "raw_value_range"):
        value = raw[key]
        if value is not None:
            setattr(info, key, int(value))
    # 0 / 65535 mean "not recorded" rather than a real calibration limit.
    if info.raw_value_range_max in (0, 65535):
        info.raw_value_range_max = None
    if info.raw_value_range_min in (0, 65535):
        info.raw_value_range_min = None

    fr = raw["frame_rate_declared"]
    if fr:
        info.frame_rate_declared = int(fr)

    ts = _parse_timestamp(blob, big)
    info.timestamp_unix_ms, info.timestamp_tz_minutes = ts
    return info


def _parse_timestamp(blob: bytes, big: bool) -> tuple[int | None, int | None]:
    off = TIMESTAMP_OFFSET
    if len(blob) < off + 10:
        return None, None
    fmt_i, fmt_h = (">I", ">h") if big else ("<I", "<h")
    seconds = struct.unpack_from(fmt_i, blob, off)[0]
    millis = struct.unpack_from(fmt_i, blob, off + 4)[0] & 0xFFFF
    tz = struct.unpack_from(fmt_h, blob, off + 8)[0]
    if seconds < 946_684_800 or seconds > 4_102_444_800:  # before 2000 or after 2100
        return None, None
    if millis >= 1000 or abs(tz) > 14 * 60:
        return None, None
    return int(seconds) * 1000 + int(millis), int(tz)
