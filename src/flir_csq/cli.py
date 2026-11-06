"""Command-line interface: ``convert_csq.py`` / ``flir-csq``.

Examples
--------
    convert_csq.py experiment_001.csq
    convert_csq.py experiment_001.csq -o results/ --format h5,csv --save-raw
    convert_csq.py experiment_001.csq --inspect
    convert_csq.py --demo /tmp/demo.csq
    convert_csq.py --verify results/experiment_001_temperature.h5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .converter import FORMATS, ConversionOptions, convert, inspect_recording
from .errors import CsqError
from .verify import verify_hdf5

PROFILE_NOTE = "FLIR FFF/CSQ container; JPEG-LS payloads need imagecodecs"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convert_csq.py",
        description="Convert a FLIR T-series .csq recording into calibrated temperature "
                    "matrices in degrees Celsius (HDF5 by default).",
        epilog="Examples:\n"
               "  convert_csq.py FLIR0001.csq\n"
               "  convert_csq.py FLIR0001.csq -o results --format h5,csv --save-raw\n"
               "  convert_csq.py FLIR0001.csq --inspect\n"
               "  convert_csq.py --demo /tmp/demo.csq\n"
               "  convert_csq.py --verify results/FLIR0001_temperature.h5\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", nargs="?", type=Path, help=".csq (or .seq) recording to convert")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="output directory (default: next to the input file)")
    p.add_argument("-f", "--format", default="h5",
                   help="output format(s), comma separated: h5, npy, csv, mat (default: h5)")
    p.add_argument("--save-raw", dest="save_raw", action="store_true", default=True,
                   help="store the decoded uint16 sensor counts (default: on)")
    p.add_argument("--no-save-raw", dest="save_raw", action="store_false",
                   help="do not store raw_counts")
    p.add_argument("--save-mask", action="store_true",
                   help="also store a uint8 valid_mask per frame")
    p.add_argument("--save-metadata", dest="save_metadata", action="store_true", default=True,
                   help="write <name>_conversion.json next to the outputs (default: on)")
    p.add_argument("--no-save-metadata", dest="save_metadata", action="store_false")
    p.add_argument("--mode", choices=("auto", "apparent", "object"), default="auto",
                   help="auto (default) = correct for emissivity/atmosphere when the recording "
                        "carries the parameters; apparent = Planck inversion only")
    p.add_argument("--raw-only", action="store_true",
                   help="export sensor counts and metadata only, even without a calibration")
    p.add_argument("--start", type=int, default=0, help="first frame index (default 0)")
    p.add_argument("--stop", type=int, default=None, help="stop before this frame index")
    p.add_argument("--stride", type=int, default=1, help="convert every Nth frame (default 1)")
    p.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    p.add_argument("--compression-level", type=int, default=4,
                   help="HDF5 gzip level 0-9 (default 4; 0 = no compression)")
    # measurement overrides
    g = p.add_argument_group("measurement overrides (replace values stored in the recording)")
    g.add_argument("--emissivity", type=float, default=None)
    g.add_argument("--distance", dest="object_distance_m", type=float, default=None,
                   help="object distance in metres")
    g.add_argument("--reflected-temperature", dest="reflected_temperature_c", type=float,
                   default=None, help="reflected apparent temperature in degC")
    g.add_argument("--atmospheric-temperature", dest="atmospheric_temperature_c", type=float,
                   default=None, help="atmospheric temperature in degC")
    g.add_argument("--humidity", dest="relative_humidity", type=float, default=None,
                   help="relative humidity, 0-1 or 0-100")
    g.add_argument("--window-transmission", type=float, default=None,
                   help="IR window transmission, 0-1")
    g.add_argument("--window-temperature", dest="window_temperature_c", type=float,
                   default=None, help="IR window temperature in degC")

    p.add_argument("--inspect", action="store_true",
                   help="print what is in the recording, convert nothing")
    p.add_argument("--verify", type=Path, metavar="H5", default=None,
                   help="verify an existing temperature .h5 file and exit")
    p.add_argument("--demo", type=Path, metavar="PATH", default=None,
                   help="write a synthetic 464x348 JPEG-LS recording and convert it")
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="machine-readable output on stdout")
    p.add_argument("-q", "--quiet", action="store_true", help="only errors")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def _formats(value: str) -> tuple[str, ...]:
    parts = tuple(dict.fromkeys(part.strip().lower() for part in value.replace(";", ",").split(",")
                                if part.strip()))
    unknown = [part for part in parts if part not in FORMATS]
    if unknown:
        raise SystemExit(f"error: unknown format(s): {', '.join(unknown)} "
                         f"(choose from {', '.join(FORMATS)})")
    return parts or ("h5",)


def _human_size(n: float | None) -> str:
    if n is None:
        return "?"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _print_inspection(info: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(info, indent=2, default=str))
        return
    cam = info["camera"]
    dims = info["dimensions"]
    timing = info["timing"]
    print(f"file        {info['path']}")
    print(f"size        {_human_size(info['file_size_bytes'])}")
    print(f"container   FLIR FFF, creator {info['container']['creator'] or '?'}, "
          f"format {info['container']['format_version']}")
    print(f"camera      {cam.get('camera_model') or 'unknown'}"
          + (f"  s/n {cam['serial_number']}" if cam.get("serial_number") else ""))
    print(f"lens        {cam.get('lens_model') or 'unknown'}"
          + (f"  FOV {cam['field_of_view_deg']:.1f} deg" if cam.get("field_of_view_deg") else ""))
    print(f"frames      {info['frames']['complete']} complete "
          f"({info['frames']['indexed']} indexed)")
    print(f"resolution  {dims['width']} x {dims['height']} px")
    print(f"payload     {info['decode']['codec']}  (record subtype {info['decode']['subtype']})")
    print(f"counts      min {info['decode']['raw_min']}  max {info['decode']['raw_max']}  "
          f"mean {info['decode']['raw_mean']}")
    if timing.get("measured_frame_rate_hz"):
        print(f"duration    {timing.get('duration_s')} s at "
              f"{timing['measured_frame_rate_hz']} Hz measured"
              + (f" (camera declares {timing['declared_frame_rate_hz']} Hz)"
                 if timing.get("declared_frame_rate_hz") else ""))
    print(f"first frame {timing.get('first_timestamp_utc') or 'no camera timestamp'}")
    print(f"radiometry  {'calibrated' if info['radiometric'] else 'NO PLANCK CALIBRATION'}"
          + (f"  (missing: {', '.join(info['missing_for_temperature'])})"
             if info["missing_for_temperature"] else ""))
    if info["radiometric"]:
        print(f"emissivity  {cam.get('emissivity')}  distance {cam.get('object_distance_m')} m  "
              f"RH {cam.get('relative_humidity')}")
        print(f"planck      R1={cam.get('planck_r1')} B={cam.get('planck_b')} F={cam.get('planck_f')} "
              f"O={cam.get('planck_o')} R2={cam.get('planck_r2')}")
    print(f"conversion  {info['temperature_conversion']}")
    for warning in info["warnings"]:
        print(f"warning     {warning}")


def _print_result(result, as_json: bool) -> None:
    if as_json:
        print(json.dumps({**result.metadata, "summary": result.as_dict()}, indent=2, default=str))
        return
    print()
    print(f"converted  {result.frames_written} frames  ->  shape {result.shape}")
    print(f"mode       {result.mode_effective}")
    agg = result.metadata.get("statistics", {}).get("aggregate", {})
    if agg.get("global_min_C") is not None:
        print(f"temperature  {agg['global_min_C']:.2f} .. {agg['global_max_C']:.2f} degC  "
              f"(mean of frame means {agg['mean_of_frame_means_C']:.2f} degC)")
    timing = result.metadata.get("timing", {})
    if timing.get("measured_frame_rate_hz"):
        print(f"timing     {timing.get('duration_s')} s at {timing['measured_frame_rate_hz']} Hz"
              + ("  [ESTIMATED]" if timing.get("timestamps_estimated") else
                 "  (camera clock)"))
    for path in result.outputs.values():
        print(f"output     {path}")
    validation = result.validation or {}
    if validation:
        print(f"verify     {'PASS' if validation.get('passed') else 'FAIL'}"
              + (f"  (max |dT| vs raw_counts {validation.get('max_recompute_difference_C'):.2e} degC)"
                 if validation.get("max_recompute_difference_C") is not None else ""))
    for warning in result.warnings[:12]:
        print(f"warning    {warning}")
    if len(result.warnings) > 12:
        print(f"warning    ... {len(result.warnings) - 12} more in the JSON report")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.verify is not None:
        report = verify_hdf5(args.verify)
        print(json.dumps(report, indent=2, default=str) if args.as_json else
              f"{'PASS' if report['passed'] else 'FAIL'}  {report['file']}\n"
              + "\n".join(f"  - {p}" for p in report.get("problems", [])))
        return 0 if report["passed"] else 3

    if args.demo is not None:
        from .synthetic import write_demo

        demo_path = args.demo
        if demo_path.is_dir() or str(args.demo).endswith("/"):
            demo_path = demo_path / "demo_t540.csq"
        truth = write_demo(demo_path)
        if not args.quiet:
            print(f"synthetic recording : {demo_path}  "
                  f"({truth['n_frames']} frames, {truth['camera'].width}x{truth['camera'].height}, "
                  f"JPEG-LS, ground truth built in)")
        args.input = demo_path
        args.overwrite = True

    if args.input is None:
        parser.print_help()
        return 2

    try:
        if args.inspect:
            info = inspect_recording(args.input)
            _print_inspection(info, args.as_json)
            return 0

        formats = _formats(args.format)
        options = ConversionOptions(
            input_path=args.input,
            output_dir=args.output,
            formats=formats,
            save_raw=args.save_raw,
            save_mask=args.save_mask,
            save_metadata=args.save_metadata,
            raw_only=args.raw_only,
            mode=args.mode,
            start=args.start,
            stop=args.stop,
            stride=args.stride,
            overwrite=args.overwrite,
            compression_level=args.compression_level,
            emissivity=args.emissivity,
            object_distance_m=args.object_distance_m,
            reflected_temperature_c=args.reflected_temperature_c,
            atmospheric_temperature_c=args.atmospheric_temperature_c,
            relative_humidity=args.relative_humidity,
            window_transmission=args.window_transmission,
            window_temperature_c=args.window_temperature_c,
        )
        last = [0]

        def progress(fraction: float, message: str, done: int, total: int) -> None:
            if args.quiet or args.as_json or not sys.stderr.isatty():
                return
            if done and done == last[0]:
                return
            last[0] = done
            sys.stderr.write(f"\r{message:60s} {fraction * 100:5.1f}%")
            sys.stderr.flush()

        options.progress = progress
        result = convert(options)
        if not sys.stderr.isatty() and not args.quiet and not args.as_json:
            sys.stderr.write("\n")
        _print_result(result, args.as_json)
        return 0
    except CsqError as exc:
        if args.as_json:
            print(json.dumps({"error": exc.message, "hint": exc.hint,
                              "detail": exc.detail}, indent=2))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return getattr(exc, "exit_code", 2)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
