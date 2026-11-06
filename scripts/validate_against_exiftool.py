#!/usr/bin/env python3
"""Cross-check a conversion against ExifTool and flirpy - an independent reference.

ExifTool is the de-facto independent reference for FLIR metadata: it is a
separate implementation, in a different language, maintained by a different
project. This script

1. reads the first frame's metadata with ExifTool and with :mod:`flir_csq`,
   and reports every value that differs;
2. extracts the first frame's raw-thermal payload with ExifTool and compares it
   byte for byte with the payload flir_csq decodes;
3. converts that payload with flirpy's ``raw2temp`` (the ThermImage port, a
   third implementation) using ExifTool's own constants, and compares the result
   with the temperature cube produced by flir_csq.

It requires ``exiftool``, ``flirpy`` and a real recording, so it is not part of
the pytest suite; it is the script used to produce ``docs/validation.md``.

    python scripts/validate_against_exiftool.py recording.csq --h5 out/recording_temperature.h5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flir_csq.container import CsqReader, scan_csq  # noqa: E402
from flir_csq.converter import inspect_recording  # noqa: E402

# ExifTool tag -> flir_csq field
TAGS = {
    "CameraModel": "camera_model",
    "CameraSerialNumber": "serial_number",
    "CameraPartNumber": "camera_part_number",
    "LensModel": "lens_model",
    "FieldOfView": "field_of_view_deg",
    "Emissivity": "emissivity",
    "ObjectDistance": "object_distance_m",
    "ReflectedApparentTemperature": "reflected_apparent_temperature_c",
    "AtmosphericTemperature": "atmospheric_temperature_c",
    "IRWindowTemperature": "ir_window_temperature_c",
    "IRWindowTransmission": "ir_window_transmission",
    "RelativeHumidity": "relative_humidity",
    "PlanckR1": "planck_r1",
    "PlanckB": "planck_b",
    "PlanckF": "planck_f",
    "PlanckO": "planck_o",
    "PlanckR2": "planck_r2",
    "RawValueRangeMin": "raw_value_range_min",
    "RawValueRangeMax": "raw_value_range_max",
    "FrameRate": "frame_rate_declared",
}


def exiftool_json(path: Path) -> dict:
    out = subprocess.run(["exiftool", "-json", "-n", str(path)], capture_output=True, text=True,
                         check=True)
    return json.loads(out.stdout)[0]


def exiftool_payload(path: Path) -> bytes:
    out = subprocess.run(["exiftool", "-b", "-RawThermalImage", str(path)],
                         capture_output=True, check=True)
    return out.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--h5", type=Path, default=None,
                        help="temperature cube produced by flir-csq (optional)")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    report: dict = {"recording": str(args.recording)}

    # 1. metadata ---------------------------------------------------------------
    et = exiftool_json(args.recording)
    info = inspect_recording(args.recording)
    camera = info["camera"]
    metadata_rows = []
    for tag, field in TAGS.items():
        if tag in et:
            metadata_rows.append({"tag": tag, "exiftool": et[tag], "flir_csq": camera.get(field),
                                  "match": _equal(et[tag], camera.get(field))})
    report["metadata"] = metadata_rows
    report["metadata_all_match"] = all(row["match"] for row in metadata_rows)

    # 2. payload identity -------------------------------------------------------
    with CsqReader(scan_csq(args.recording)) as reader:
        ours_payload = _raw_payload_bytes(reader, 0)
        reference_payload = exiftool_payload(args.recording)
    report["payload"] = {
        "exiftool_sha256": hashlib.sha256(reference_payload).hexdigest(),
        "flir_csq_sha256": hashlib.sha256(ours_payload).hexdigest(),
        "identical": ours_payload == reference_payload,
        "bytes": len(ours_payload),
    }

    # 3. independent temperature conversion -------------------------------------
    raw = None
    try:
        import imagecodecs

        raw = imagecodecs.jpegls_decode(reference_payload)
    except Exception as exc:  # noqa: BLE001 - payload may not be JPEG-LS
        report["temperature"] = {"skipped": f"cannot decode the ExifTool payload: {exc}"}

    if args.h5 is not None and raw is not None:
        import h5py
        from flirpy.util.raw import raw2temp

        meta = {
            "Atmospheric Trans Alpha 1": "0.006569",
            "Atmospheric Trans Alpha 2": "0.012620",
            "Atmospheric Trans Beta 1": "-0.002276",
            "Atmospheric Trans Beta 2": "-0.006670",
            "Atmospheric Trans X": "1.900000",
            "Planck R1": str(et["PlanckR1"]), "Planck R2": str(et["PlanckR2"]),
            "Planck O": str(et["PlanckO"]), "Planck B": str(et["PlanckB"]),
            "Planck F": str(et["PlanckF"]),
            "Emissivity": str(et["Emissivity"]),
            "IR Window Transmission": str(et.get("IRWindowTransmission", 1.0)),
            "IR Window Temperature": str(et.get("IRWindowTemperature", 20.0)),
            "Object Distance": str(et.get("ObjectDistance", 1.0)),
            "Atmospheric Temperature": str(et.get("AtmosphericTemperature", 20.0)),
            "Reflected Apparent Temperature": str(et.get("ReflectedApparentTemperature", 20.0)),
            # ExifTool reports the FLIR field as a 0-1 fraction, which is what flirpy expects.
            "Relative Humidity": str(et.get("RelativeHumidity", 0.5)),
        }
        reference_t = raw2temp(raw.astype(np.uint16), meta)
        with h5py.File(args.h5, "r") as f:
            ours_t = np.asarray(f["temperature_C"][0], dtype=np.float64)
        diff = ours_t - reference_t
        report["temperature"] = {
            "reference": "flirpy raw2temp (ThermImage port) with ExifTool constants",
            "reference_min_C": float(np.nanmin(reference_t)),
            "reference_max_C": float(np.nanmax(reference_t)),
            "flir_csq_min_C": float(np.nanmin(ours_t)),
            "flir_csq_max_C": float(np.nanmax(ours_t)),
            "max_abs_diff_C": float(np.nanmax(np.abs(diff))),
            "mean_abs_diff_C": float(np.nanmean(np.abs(diff))),
            "note": "flirpy applies a two-half-path atmosphere with both halves at "
                    f"{et.get('ObjectDistance')} m; flir-csq applies one path of that length, "
                    "so a small residual is expected when transmission < 1",
        }

    print(json.dumps(report, indent=2, default=str))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0


def _equal(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-3 * max(1.0, abs(float(a)))
    return str(a).strip() == str(b).strip()


def _raw_payload_bytes(reader: CsqReader, i: int) -> bytes:
    from flir_csq.fff import RAW_DATA, find_record, parse_header, parse_records

    data = reader.frame_bytes(i)
    header = parse_header(data, 0)
    records = parse_records(data, 0, header)
    rec = find_record(records, RAW_DATA)
    assert rec is not None
    return data[rec.offset:rec.offset + rec.length][0x20:]


if __name__ == "__main__":
    raise SystemExit(main())
