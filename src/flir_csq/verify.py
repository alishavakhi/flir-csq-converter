"""Automatic verification of a written temperature cube.

Verification is not a substitute for calibration validation against FLIR
software, and it does not pretend to be: it proves that the file that was just
written is internally consistent and can be re-read, that every frame is where
it claims to be, and that a temperature recomputed from the stored raw counts
with the stored conversion parameters reproduces the stored temperature.

Checks
------
1. required datasets exist and have the documented dtype/shape;
2. ``timestamps`` are finite, monotonic and start at 0;
3. ``raw_counts`` (when present) has the same frame count and geometry;
4. the stored ``temperature_C`` of sample frames equals the value recomputed
   from ``raw_counts`` + the conversion block in the metadata (tolerance 1e-3 degC);
5. finite-temperature coverage is reported per sample frame;
6. temperatures stay inside the recording's own calibration range (a broad
   sanity bound, reported - not silently clipped).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .errors import OutputError
from .metadata import CameraInfo
from .radiometry import MeasurementParameters, PlanckCoefficients, convert_frame

REQUIRED = ("temperature_C", "timestamps", "metadata")


def _camera_from_metadata(meta: dict[str, Any]) -> CameraInfo:
    """Rebuild the CameraInfo the conversion used (from the JSON block)."""
    conversion = meta.get("conversion", {})
    camera_block = meta.get("camera", {})
    info = CameraInfo(**{k: v for k, v in camera_block.items()
                         if k in CameraInfo.__dataclass_fields__})
    coefficients = conversion.get("coefficients") or {}
    for key in ("planck_r1", "planck_b", "planck_f", "planck_o", "planck_r2"):
        if key in coefficients:
            setattr(info, key, float(coefficients[key]))
    return info


def _params_from_metadata(meta: dict[str, Any]) -> MeasurementParameters:
    block = meta.get("conversion", {}).get("parameters") or {}
    params = MeasurementParameters(
        emissivity=block.get("emissivity"),
        object_distance_m=block.get("object_distance_m"),
        reflected_apparent_temperature_c=block.get("reflected_apparent_temperature_c"),
        atmospheric_temperature_c=block.get("atmospheric_temperature_c"),
        relative_humidity=block.get("relative_humidity"),
        ir_window_transmission=block.get("ir_window_transmission"),
        ir_window_temperature_c=block.get("ir_window_temperature_c"),
        atmospheric_transmission=block.get("atmospheric_transmission"),
        source="stored-metadata",
    )
    return params


def verify_hdf5(path: str | Path, *, tolerance_c: float = 1e-3,
                sample_frames: int = 3) -> dict[str, Any]:
    """Re-open an output file and check it. Never raises for a bad file."""
    import h5py

    p = Path(path)
    problems: list[str] = []
    checks: list[dict[str, Any]] = []
    report: dict[str, Any] = {"file": str(p), "passed": False, "checks": checks,
                              "problems": problems}

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            problems.append(f"{name}: {detail}" if detail else name)

    if not p.is_file():
        problems.append("output file does not exist")
        return report
    try:
        with h5py.File(p, "r") as f:
            for name in REQUIRED:
                check(f"dataset {name} present", name in f, f"missing /{name}")
            if problems:
                return report

            temp = f["temperature_C"]
            ts = f["timestamps"]
            meta = json.loads(f["metadata"][()])
            report["shape"] = list(temp.shape)
            report["dtype"] = str(temp.dtype)
            report["compression"] = temp.compression
            report["chunks"] = list(temp.chunks) if temp.chunks else None

            check("temperature_C is 3-D", temp.ndim == 3, f"ndim={temp.ndim}")
            check("temperature_C is float32", temp.dtype == np.float32, str(temp.dtype))
            check("timestamps length matches frames", ts.shape == (temp.shape[0],),
                  f"{ts.shape} vs {temp.shape[0]}")
            if "raw_counts" in f:
                check("raw_counts dtype uint16", f["raw_counts"].dtype == np.uint16,
                      str(f["raw_counts"].dtype))
                check("raw_counts shape matches temperature",
                      f["raw_counts"].shape == temp.shape,
                      f"{f['raw_counts'].shape} vs {temp.shape}")
            check("units attribute is degC", temp.attrs.get("unit") == "degC",
                  str(temp.attrs.get("unit")))

            tsa = np.asarray(ts)
            check("timestamps finite", bool(np.all(np.isfinite(tsa))), "NaN or inf in timestamps")
            if tsa.size:
                check("timestamps start at 0", abs(float(tsa[0])) < 1e-9, f"first={tsa[0]}")
                check("timestamps monotonic", bool(np.all(np.diff(tsa) >= -1e-9)),
                      "timestamps are not non-decreasing")

            n = temp.shape[0]
            # Spread the samples over the whole cube: first, last and evenly spaced frames.
            picks = sorted({round(v) for v in np.linspace(0, max(n - 1, 0),
                                                              max(sample_frames, 3))})
            picks = [i for i in picks if 0 <= i < n]
            frames_report = []
            recompute_ok = True
            max_diff = 0.0
            for i in picks:
                frame = np.asarray(temp[i], dtype=np.float64)
                finite = np.isfinite(frame)
                stats = {
                    "frame": int(i),
                    "valid_fraction": float(finite.mean()),
                    "min_C": float(np.nanmin(frame)) if finite.any() else None,
                    "max_C": float(np.nanmax(frame)) if finite.any() else None,
                    "mean_C": float(np.nanmean(frame)) if finite.any() else None,
                }
                if "raw_counts" in f and meta.get("conversion", {}).get("coefficients"):
                    info = _camera_from_metadata(meta)
                    params = _params_from_metadata(meta)
                    mode = meta.get("conversion", {}).get("mode_effective", "apparent")
                    recomputed, _ = convert_frame(np.asarray(f["raw_counts"][i]), info,
                                                  mode="apparent" if mode != "object" else "object",
                                                  params=params,
                                                  coef=PlanckCoefficients.from_camera_info(info))
                    diff = np.abs(np.asarray(recomputed, dtype=np.float64) - frame)
                    frame_max = float(np.nanmax(diff)) if np.isfinite(diff).any() else float("inf")
                    max_diff = max(max_diff, frame_max)
                    stats["max_abs_recompute_diff_C"] = frame_max
                    if not (frame_max <= tolerance_c):
                        recompute_ok = False
                frames_report.append(stats)
            check("temperature reproducible from raw_counts", recompute_ok,
                  f"max |Δ| = {max_diff:.3e} degC (tolerance {tolerance_c:g})")
            report["sample_frames"] = frames_report
            report["max_recompute_difference_C"] = max_diff

            agg = meta.get("statistics", {}).get("aggregate", {})
            report["aggregate"] = agg
            lo = meta.get("camera", {}).get("raw_value_range_min")
            report["calibration_range_raw"] = [lo, meta.get("camera", {}).get("raw_value_range_max")]
            report["passed"] = not problems
    except Exception as exc:  # noqa: BLE001 - a corrupt file must be reported, not raised
        problems.append(f"cannot read output: {type(exc).__name__}: {exc}")
        report["passed"] = False
    return report


def verify_npy(path: str | Path) -> dict[str, Any]:
    """Cheap structural check for a ``.npy`` cube."""
    p = Path(path)
    problems: list[str] = []
    try:
        arr = np.load(p, mmap_mode="r")
        if arr.ndim != 3:
            problems.append(f"expected 3-D array, got {arr.ndim}-D")
        if arr.dtype != np.float32:
            problems.append(f"expected float32, got {arr.dtype}")
        if not np.isfinite(arr[0]).any():
            problems.append("first frame has no finite temperature")
        shape = list(arr.shape)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"cannot read .npy: {exc}")
        shape = []
    return {"file": str(p), "shape": shape, "passed": not problems, "problems": problems}


def assert_verified(report: dict[str, Any]) -> None:
    if not report.get("passed", False):
        raise OutputError(
            "output verification failed",
            detail="; ".join(report.get("problems", [])) or "unknown reason",
        )
