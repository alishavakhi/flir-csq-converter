#!/usr/bin/env python3
"""Profile where memory goes during a long conversion.

Runs one stage at a time over the same recording and prints RSS every N frames, so
decode, radiometry and the writers can be separated instead of guessed about. This
is the script used for the measurements in ``docs/troubleshooting.md``.

Examples::

    # build a synthetic 3000-frame T540-sized recording (one frame at a time)
    python scripts/profile_memory.py build --frames 3000

    # decode only / decode + radiometry / full conversion
    python scripts/profile_memory.py decode  --recording .scratch/data/long.csq
    python scripts/profile_memory.py convert --recording .scratch/data/long.csq
    python scripts/profile_memory.py h5      --recording .scratch/data/long.csq --frames 300
    python scripts/profile_memory.py mat73   --recording .scratch/data/long.csq --frames 300
"""

from __future__ import annotations

import argparse
import gc
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flir_csq.container import CsqReader, scan_csq  # noqa: E402
from flir_csq.radiometry import convert_frame  # noqa: E402
from flir_csq.synthetic import make_streaming_csq  # noqa: E402


def rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def report(stage: str, i: int, every: int, base: float, t0: float) -> None:
    if not i or i % every:
        return
    gc.collect()
    rss = rss_mb()
    print(f"  {stage:8s} frame {i:6d}  rss {rss:8.1f} MB  (+{rss - base:7.1f})  "
          f"peak {peak_mb():8.1f}  {(time.perf_counter() - t0) / i * 1000:6.2f} ms/frame",
          flush=True)


def stage_decode(recording: Path, frames: int, every: int) -> None:
    index = scan_csq(recording)
    n = min(frames or len(index), len(index))
    base, t0 = rss_mb(), time.perf_counter()
    with CsqReader(index) as reader:
        for i in range(n):
            reader.read_frame(i)
            report("decode", i + 1, every, base, t0)
    print(f"decode done: +{rss_mb() - base:.1f} MB, peak {peak_mb():.1f} MB")


def stage_convert(recording: Path, frames: int, every: int) -> None:
    index = scan_csq(recording)
    n = min(frames or len(index), len(index))
    base, t0 = rss_mb(), time.perf_counter()
    with CsqReader(index) as reader:
        for i in range(n):
            frame = reader.read_frame(i)
            convert_frame(frame.raw, reader.camera_info(i), mode="auto")
            report("convert", i + 1, every, base, t0)
    print(f"convert done: +{rss_mb() - base:.1f} MB, peak {peak_mb():.1f} MB")


def stage_writer(recording: Path, fmt: str, frames: int, every: int, out_dir: Path) -> None:
    from flir_csq.converter import ConversionOptions, convert

    base, t0 = rss_mb(), time.perf_counter()

    def progress(fraction: float, message: str, done: int, total: int) -> None:
        report(fmt, done, every, base, t0)

    convert(ConversionOptions(input_path=recording, output_dir=out_dir, formats=(fmt,),
                              mat_version="7.3", overwrite=True, save_metadata=False,
                              save_raw=False,
                              stop=frames if frames else None, progress=progress))
    size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"{fmt} done: +{rss_mb() - base:.1f} MB, peak {peak_mb():.1f} MB, "
          f"output {size / 1e6:.0f} MB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("build", "decode", "convert", "h5", "mat73"))
    parser.add_argument("--recording", type=Path,
                        default=ROOT / ".scratch" / "data" / "long.csq")
    parser.add_argument("--frames", type=int, default=0,
                        help="frames to build/process (0 = whole recording)")
    parser.add_argument("--every", type=int, default=200, help="report interval in frames")
    parser.add_argument("--height", type=int, default=348)
    parser.add_argument("--width", type=int, default=464)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.stage == "build":
        n = args.frames or 1000
        args.recording.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        make_streaming_csq(args.recording, n_frames=n, height=args.height, width=args.width)
        print(f"built {n} frames in {time.perf_counter() - t0:.0f} s, "
              f"{args.recording.stat().st_size / 1e6:.0f} MB -> {args.recording}")
        return 0

    if not args.recording.is_file():
        print(f"recording not found: {args.recording}\nrun the build stage first",
              file=sys.stderr)
        return 2
    out = args.out or args.recording.parent / f"out_{args.stage}"
    if args.stage == "decode":
        stage_decode(args.recording, args.frames, args.every)
    elif args.stage == "convert":
        stage_convert(args.recording, args.frames, args.every)
    else:
        stage_writer(args.recording, "mat" if args.stage == "mat73" else "h5",
                     args.frames, args.every, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
