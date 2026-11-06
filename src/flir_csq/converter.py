"""Conversion job: CSQ recording -> temperature cube + metadata + report.

The job is sequential and bounded: one frame is read, decoded, converted and
written before the next is touched, so a 20 000-frame recording needs about one
frame of memory. Every converted frame is written in recording order and its
index is recorded in the report, so a gap can never be silent.

Outputs are staged (``<name>.partial``) and renamed only after the last frame is
written and the file is flushed: an interrupted or cancelled run leaves no
half-written cube that looks complete.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import platform
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .codecs import available_codecs, can_decode_jpegls
from .container import CsqReader, scan_csq
from .errors import (
    ConversionCancelled,
    CsqError,
    MalformedFrame,
    MissingCalibration,
    NotACsqFile,
    OutputError,
)
from .exporter import (
    CONVERTER_VERSION,
    default_output_path,
    make_exporter,
    rewrite_hdf5_metadata,
)
from .metadata import CameraInfo
from .radiometry import (
    MODES,
    ConversionInfo,
    MeasurementParameters,
    atmospheric_transmission,
    convert_frame,
    frame_statistics,
    measurement_parameters_from,
)

LOG = logging.getLogger("flir_csq")

ProgressCallback = Callable[[float, str, int, int], None]
CancelCheck = Callable[[], bool]

FORMATS = ("h5", "npy", "csv", "mat")


@dataclass
class ConversionOptions:
    input_path: Path
    output_dir: Path | None = None
    formats: tuple[str, ...] = ("h5",)
    save_raw: bool = True
    save_mask: bool = False
    save_metadata: bool = True
    raw_only: bool = False
    mode: str = "auto"
    start: int = 0
    stop: int | None = None
    stride: int = 1
    overwrite: bool = False
    compression_level: int = 4
    mat_version: str = "auto"          # auto | 7 | 7.3 (MATLAB output container)
    mat73_threshold_bytes: int = 1_800_000_000
    # measurement overrides (recorded as overrides in the output metadata)
    emissivity: float | None = None
    object_distance_m: float | None = None
    reflected_temperature_c: float | None = None
    atmospheric_temperature_c: float | None = None
    relative_humidity: float | None = None
    window_transmission: float | None = None
    window_temperature_c: float | None = None
    progress: ProgressCallback | None = None
    should_cancel: CancelCheck | None = None

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        if self.mode not in MODES and not self.raw_only:
            raise ValueError(f"mode must be one of {MODES}")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.mat_version not in ("auto", "7", "7.3", "v7", "v7.3", "5", "73"):
            raise ValueError("mat_version must be auto, 7 or 7.3")
        if self.start < 0:
            raise ValueError("start must be >= 0")
        if self.stop is not None and self.stop <= self.start:
            raise ValueError("stop must be greater than start")
        unknown = [f for f in self.formats if f not in FORMATS]
        if unknown:
            raise ValueError(f"unknown output format(s): {', '.join(unknown)}")


@dataclass
class ConversionResult:
    input_path: Path
    outputs: dict[str, Path] = field(default_factory=dict)
    frames_written: int = 0
    frames_requested: int = 0
    n_frames_in_file: int = 0
    shape: tuple[int, int, int] = (0, 0, 0)
    mode_effective: str = ""
    status: str = "ok"  # ok | raw-only | unsupported
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def primary_output(self) -> Path | None:
        for fmt in ("h5", "npy", "mat", "csv"):
            if fmt in self.outputs:
                return self.outputs[fmt]
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "input": str(self.input_path),
            "outputs": {k: str(v) for k, v in self.outputs.items()},
            "frames_written": self.frames_written,
            "frames_requested": self.frames_requested,
            "n_frames_in_file": self.n_frames_in_file,
            "shape": list(self.shape),
            "mode_effective": self.mode_effective,
            "status": self.status,
            "warnings": list(self.warnings),
            "validation": self.validation,
            "duration_s": round(self.duration_s, 3),
        }


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def _fingerprint(path: Path) -> dict[str, Any]:
    """Cheap, stable identity for reproducibility notes (no full-file hash)."""
    stat = path.stat()
    with path.open("rb") as fh:
        head = fh.read(1 << 16)
        tail_size = min(1 << 16, stat.st_size)
        fh.seek(max(0, stat.st_size - tail_size))
        tail = fh.read(tail_size)
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256_head_64KiB": hashlib.sha256(head).hexdigest(),
        "sha256_tail_64KiB": hashlib.sha256(tail).hexdigest(),
    }


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _missing_calibration(info: CameraInfo) -> list[str]:
    return [name for name in ("planck_r1", "planck_b", "planck_f", "planck_o", "planck_r2")
            if getattr(info, name) is None]


def _staging_path(target: Path) -> Path:
    return target.with_name(target.name + ".partial")


def _apply_overrides(info: CameraInfo,
                     options: ConversionOptions) -> tuple[MeasurementParameters, list[str]]:
    """Measurement parameters for a frame, with any explicit CLI/GUI overrides."""
    params = measurement_parameters_from(info)
    overrides = {
        "emissivity": options.emissivity,
        "object_distance_m": options.object_distance_m,
        "reflected_apparent_temperature_c": options.reflected_temperature_c,
        "atmospheric_temperature_c": options.atmospheric_temperature_c,
        "relative_humidity": options.relative_humidity,
        "ir_window_transmission": options.window_transmission,
        "ir_window_temperature_c": options.window_temperature_c,
    }
    notes = [f"{name} overridden to {value}" for name, value in overrides.items()
             if value is not None]
    for name, value in overrides.items():
        if value is not None:
            setattr(params, name, value)
    if notes:
        params.source = "file-metadata+override"
        if params.object_distance_m is not None and params.atmospheric_temperature_c is not None \
                and params.relative_humidity is not None:
            params.atmospheric_transmission = atmospheric_transmission(
                float(params.object_distance_m), float(params.atmospheric_temperature_c),
                float(params.relative_humidity))
    return params, notes


def _aggregate(per_frame: list[dict[str, Any]], raw_only: bool) -> dict[str, Any]:
    if raw_only:
        return {"n_frames": len(per_frame), "note": "raw-only export: no temperature statistics"}
    temps = [f for f in per_frame if "min_C" in f]
    mins = [f["min_C"] for f in temps if f["min_C"] is not None]
    maxs = [f["max_C"] for f in temps if f["max_C"] is not None]
    means = [f["mean_C"] for f in temps if f["mean_C"] is not None]
    valid = [f["valid_fraction"] for f in temps]
    return {
        "n_frames": len(temps),
        "global_min_C": min(mins) if mins else None,
        "global_max_C": max(maxs) if maxs else None,
        "mean_of_frame_means_C": sum(means) / len(means) if means else None,
        "mean_valid_fraction": sum(valid) / len(valid) if valid else 0.0,
        "frames_with_no_valid_pixels": sum(1 for f in temps if (f.get("n_valid") or 0) == 0),
    }


# ----------------------------------------------------------------------------------
# inspection
# ----------------------------------------------------------------------------------
def inspect_recording(path: str | Path) -> dict[str, Any]:
    """Read everything worth knowing about a recording without converting it."""
    p = Path(path)
    index = scan_csq(p)
    reader = CsqReader(index)
    warnings = list(index.warnings)
    try:
        complete = [i for i, f in enumerate(index.frames) if not f.truncated]
        if not complete:
            raise NotACsqFile("every frame in this file is incomplete")
        first = complete[0]
        frame = reader.read_frame(first)
        info = reader.camera_info(first)
        warnings.extend(frame.warnings)
        missing = _missing_calibration(info)
        timing = _timing_from_reader(reader, complete)
        return {
            "path": str(p.resolve()),
            "file_size_bytes": index.file_size,
            "container": index.as_dict(),
            "frames": {"indexed": index.n_frames, "complete": index.complete_frames,
                       "truncated": index.n_frames - index.complete_frames},
            "dimensions": {"height": frame.height, "width": frame.width,
                           "pixels": frame.height * frame.width},
            "decode": {"codec": frame.codec, "subtype": frame.subtype,
                       "raw_min": int(frame.raw.min()), "raw_max": int(frame.raw.max()),
                       "raw_mean": round(float(frame.raw.mean()), 2)},
            "camera": info.as_dict(),
            "timing": timing,
            "decoders": [c.as_dict() for c in available_codecs()],
            "jpeg_ls_supported": can_decode_jpegls(),
            "radiometric": not missing,
            "temperature_conversion": "supported" if not missing else "unsupported",
            "missing_for_temperature": missing,
            "warnings": _dedupe(warnings),
        }
    finally:
        reader.close()


def _timing_from_reader(reader: CsqReader, frames: list[int]) -> dict[str, Any]:
    """Timing summary from the camera clock of every frame (headers only)."""
    stamps: list[int | None] = []
    declared: int | None = None
    tz: int | None = None
    for i in frames:
        info = reader.camera_info(i)
        stamps.append(info.timestamp_unix_ms)
        declared = declared or info.frame_rate_declared
        if tz is None and info.timestamp_tz_minutes is not None:
            tz = info.timestamp_tz_minutes
    have = [s for s in stamps if s is not None]
    out: dict[str, Any] = {
        "declared_frame_rate_hz": declared,
        "n_frames": len(frames),
        "n_timestamps_available": len(have),
        "timezone_offset_minutes": tz,
        "timestamps_estimated": True,
        "timestamp_source": "estimated from the declared frame rate: the recording carries no "
                            "camera timestamps",
    }
    if len(have) == len(frames) and len(frames) >= 2:
        span = (have[-1] - have[0]) / 1000.0
        out.update({
            "timestamp_source": "camera clock (DateTimeOriginal, millisecond resolution)",
            "timestamps_estimated": False,
            "first_timestamp_utc": _dt.datetime.fromtimestamp(
                have[0] / 1000.0, tz=_dt.timezone.utc).isoformat(),
            "last_timestamp_utc": _dt.datetime.fromtimestamp(
                have[-1] / 1000.0, tz=_dt.timezone.utc).isoformat(),
            "duration_s": round(span, 6),
            "measured_frame_rate_hz": round((len(frames) - 1) / span, 6) if span > 0 else None,
        })
    elif have:
        out["first_timestamp_utc"] = _dt.datetime.fromtimestamp(
            have[0] / 1000.0, tz=_dt.timezone.utc).isoformat()
        out["timestamp_source"] = "camera clock, only partially present"
    return out


# ----------------------------------------------------------------------------------
# conversion
# ----------------------------------------------------------------------------------
def convert(options: ConversionOptions) -> ConversionResult:
    """Run a conversion. Raises :class:`CsqError` subclasses on handled failures."""
    t_start = time.perf_counter()
    progress = options.progress or (lambda *_: None)
    cancelled = options.should_cancel or (lambda: False)
    input_path = options.input_path
    if not input_path.is_file():
        raise NotACsqFile(f"file not found: {input_path}")

    progress(0.0, "indexing recording", 0, 0)
    index = scan_csq(input_path)
    warnings: list[str] = list(index.warnings)
    n_complete = index.complete_frames
    if n_complete < index.n_frames:
        warnings.append(
            f"{index.n_frames - n_complete} incomplete frame(s) at the end of the recording "
            "were not converted"
        )

    stop = min(options.stop if options.stop is not None else n_complete, n_complete)
    frames = list(range(options.start, stop, options.stride))
    if not frames:
        raise CsqError(
            f"no frames selected (file has {n_complete} complete frames; requested "
            f"{options.start}..{stop} step {options.stride})"
        )

    reader = CsqReader(index)
    exporters: dict[str, Any] = {}
    targets: dict[str, Path] = {}
    staged: dict[str, Path] = {}
    published: list[Path] = []
    mat_container = ""
    try:
        first_info = reader.camera_info(frames[0])
        missing = _missing_calibration(first_info)
        raw_only = options.raw_only
        if missing and not raw_only:
            raise MissingCalibration(
                "this recording carries no usable Planck calibration, so no temperature can be "
                "computed (missing: " + ", ".join(missing) + ")",
                hint="Re-run with --raw-only to export the radiometric counts and metadata "
                     "exactly as stored, or re-record with the camera calibration intact.",
            )
        if raw_only:
            warnings.append("raw-only export: no temperature was computed for this recording")

        probe = reader.read_frame(frames[0])
        height, width = probe.shape

        for fmt in options.formats:
            target = default_output_path(input_path, options.output_dir, fmt)
            if target.exists() and not options.overwrite:
                raise OutputError(
                    f"output already exists: {target}",
                    hint="Pass --overwrite (or tick 'overwrite' in the GUI) to replace it.",
                )
            staging = _staging_path(target)
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True) if staging.is_dir() else staging.unlink()
            exporter = make_exporter(fmt, staging, compression_level=options.compression_level,
                                     atomic=False, mat_version=options.mat_version,
                                     mat73_threshold_bytes=options.mat73_threshold_bytes)
            exporter.open(n_frames=len(frames), height=height, width=width,
                          save_raw=options.save_raw, save_mask=options.save_mask,
                          temperature=not raw_only)
            exporters[fmt] = exporter
            targets[fmt] = target
            staged[fmt] = staging
            if fmt == "mat":
                mat_container = getattr(exporter, "used_version", "")

        total = len(frames)
        t0_ms = first_info.timestamp_unix_ms
        fallback_fps = float(first_info.frame_rate_declared or 30)
        per_frame: list[dict[str, Any]] = []
        converted_info: ConversionInfo | None = None
        warnings_acc: list[str] = []
        previous_shape: tuple[int, int] | None = None
        previous_calibration: tuple | None = None
        calibration_changes = 0
        timestamps_estimated = False

        for out_index, frame_index in enumerate(frames):
            if cancelled():
                for exporter in exporters.values():
                    exporter.abort()
                raise ConversionCancelled(
                    f"conversion cancelled after {out_index} of {total} frames",
                    hint="No partial output was kept.",
                )
            info = first_info if frame_index == frames[0] else reader.camera_info(frame_index)
            raw_frame = reader.read_frame(frame_index)
            if previous_shape is not None and raw_frame.shape != previous_shape:
                raise MalformedFrame(
                    f"frame {frame_index} is {raw_frame.shape[1]}x{raw_frame.shape[0]}, but "
                    f"earlier frames are {previous_shape[1]}x{previous_shape[0]}",
                    hint="The recording changes geometry mid-file; convert each part separately.",
                )
            previous_shape = raw_frame.shape
            calibration = info.calibration_signature()
            if previous_calibration is not None and calibration != previous_calibration:
                calibration_changes += 1
            previous_calibration = calibration
            warnings_acc.extend(f"frame {frame_index}: {w}" for w in raw_frame.warnings)

            temperature: np.ndarray | None = None
            stats: dict[str, Any]
            if raw_only:
                stats = {"frame_index": frame_index}
            else:
                params, notes = _apply_overrides(info, options)
                warnings_acc.extend(notes)
                temperature, converted_info = convert_frame(
                    raw_frame.raw, info, mode=options.mode, params=params)
                warnings_acc.extend(converted_info.warnings)
                stats = {"frame_index": frame_index, **frame_statistics(temperature)}

            ms = info.timestamp_unix_ms
            if ms is not None and t0_ms is not None:
                ts = (ms - t0_ms) / 1000.0
            else:
                ts = out_index / fallback_fps
                timestamps_estimated = True
            stats.update({
                "output_index": out_index,
                "timestamp_s": round(float(ts), 6),
                "timestamp_unix_ms": int(ms) if ms is not None else None,
                "raw_min": int(raw_frame.raw.min()),
                "raw_max": int(raw_frame.raw.max()),
                "codec": raw_frame.codec,
            })
            per_frame.append(stats)

            for exporter in exporters.values():
                exporter.write(out_index, temperature, raw_frame.raw, ts, ms)
            if out_index % 10 == 0 or out_index == total - 1:
                progress((out_index + 1) / total,
                         f"frame {frame_index} ({out_index + 1}/{total})",
                         out_index + 1, total)

        timing = _timing_from_reader(reader, frames)
        if timestamps_estimated:
            timing.update({
                "timestamps_estimated": True,
                "timestamp_source": "estimated: frame index / declared frame rate (the recording "
                                    "carries no usable camera timestamps)",
            })
        conversion_meta: dict[str, Any] = {
            "mode_requested": options.mode,
            "mode_effective": ("raw-only" if raw_only else
                               (converted_info.mode_effective if converted_info else options.mode)),
            "corrections_applied": bool(converted_info and converted_info.mode_effective == "object"),
            "units": "degC",
            "method": (
                "none (raw counts exported)" if raw_only else
                "Planck inversion T = B / ln(R1/(R2*(raw+O)) + F) - 273.15"
                + (" with emissivity, reflected-apparent-temperature, atmospheric-transmission "
                   "and IR-window correction"
                   if converted_info and converted_info.mode_effective == "object" else
                   " (no emissivity/atmosphere correction)")
            ),
            "coefficients": converted_info.coefficients.as_dict() if converted_info else None,
            "parameters": converted_info.parameters.as_dict() if converted_info else None,
            "atmospheric_transmission": (converted_info.atmospheric_transmission
                                         if converted_info else None),
            "decoder": probe.codec,
            "matlab_container": mat_container or None,
            "per_frame_calibration_changes": calibration_changes,
            "invalid_pixels_note": (
                "pixels outside the recording's raw-value range, or at the 16-bit rail, are NaN"
            ),
        }

        created = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
        metadata: dict[str, Any] = {
            "converter": {
                "name": "flir-csq",
                "version": CONVERTER_VERSION,
                "decoder": "FLIR FFF container + per-frame payload codec (see decoder_details)",
                "decoder_details": [c.as_dict() for c in available_codecs()],
                "python": platform.python_version(),
                "platform": platform.platform(),
                "created_utc": created,
                "command": " ".join(_argv()),
            },
            "recording": {
                "path": str(input_path.resolve()),
                "fingerprint": _fingerprint(input_path),
                "container": index.as_dict(),
                "n_frames_indexed": index.n_frames,
                "n_frames_converted": total,
                "converted_frame_indices": frames,
                "start": options.start,
                "stop": stop,
                "stride": options.stride,
            },
            "camera": first_info.as_dict(),
            "geometry": {
                "height": height,
                "width": width,
                "pixels_per_frame": height * width,
                "shape": [total, height, width],
                "note": "dimensions are taken from the recording, never assumed",
            },
            "timing": timing,
            "conversion": conversion_meta,
            "statistics": {"per_frame": per_frame, "aggregate": _aggregate(per_frame, raw_only)},
            "outputs": {},
            "warnings": [],
            "validation": {},
        }
        # Output paths are known before finalising; sizes are filled in afterwards.
        outputs_paths = {fmt: targets[fmt] for fmt in exporters}
        metadata["outputs"] = {
            fmt: {"path": str(path), "size_bytes": None} for fmt, path in outputs_paths.items()
        }
        if options.save_metadata:
            report_path = default_output_path(input_path, options.output_dir, "h5").with_name(
                f"{input_path.stem}_conversion.json")
            metadata["outputs"]["json"] = {"path": str(report_path), "size_bytes": None}

        for fmt, exporter in exporters.items():
            exporter.finalize(metadata)
            target = targets[fmt]
            if target.exists() and target.is_dir():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            staged[fmt].replace(target)
            published.append(target)

        result = ConversionResult(
            input_path=input_path,
            outputs={fmt: targets[fmt] for fmt in exporters},
            frames_written=total,
            frames_requested=total,
            n_frames_in_file=index.n_frames,
            shape=(total, height, width),
            mode_effective=conversion_meta["mode_effective"],
            status="raw-only" if raw_only else "ok",
            metadata=metadata,
            duration_s=time.perf_counter() - t_start,
        )

        # Sizes, warnings and the verification result only exist now.
        for fmt, path in list(result.outputs.items()):
            metadata["outputs"][fmt] = {
                "path": str(path),
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        metadata["warnings"] = _dedupe(warnings + warnings_acc)
        result.warnings = metadata["warnings"]

        if "h5" in result.outputs:
            from .verify import verify_hdf5

            result.validation = verify_hdf5(result.outputs["h5"])
            metadata["validation"] = result.validation
            if not result.validation.get("passed", False):
                result.warnings.extend(result.validation.get("problems", []))
                metadata["warnings"] = result.warnings
        elif "mat" in result.outputs:
            from .verify import verify_mat

            result.validation = verify_mat(result.outputs["mat"])
            metadata["validation"] = result.validation
            if not result.validation.get("passed", False):
                result.warnings.extend(result.validation.get("problems", []))
                metadata["warnings"] = result.warnings

        if options.save_metadata:
            metadata["outputs"]["json"] = {"path": str(report_path), "size_bytes": None}
            result.outputs["json"] = report_path
            report_path.write_text(
                json.dumps({**metadata, "summary": result.as_dict()},
                           indent=2, default=str, allow_nan=True),
                encoding="utf-8")
            metadata["outputs"]["json"]["size_bytes"] = report_path.stat().st_size
        if "h5" in result.outputs:
            rewrite_hdf5_metadata(result.outputs["h5"], metadata)
        return result

    except BaseException as exc:
        for exporter in exporters.values():
            try:
                exporter.abort()
            except Exception:  # noqa: BLE001,S110
                pass
        for path in list(staged.values()) + published:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            except OSError:
                pass
        if isinstance(exc, KeyboardInterrupt):
            raise ConversionCancelled("conversion interrupted by the user") from exc
        raise
    finally:
        reader.close()
        for exporter in exporters.values():
            try:
                exporter.abort()
            except Exception:  # noqa: BLE001,S110 - abort after success is a no-op
                pass


def _argv() -> list[str]:
    import sys

    return sys.argv
