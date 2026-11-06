#!/usr/bin/env python3
"""Analyse a temperature cube: heatmap, one pixel over time, mean frame temperature.

    python examples/analyze_temperatures.py experiment_001_temperature.h5 --out ./figures

Three figures are written:

* ``heatmap_frame_<n>.png`` - the thermal map of one frame in degC;
* ``pixel_timeseries.png`` - temperature of one pixel versus time;
* ``mean_temperature.png`` - mean and maximum frame temperature versus time.

The plots are conveniences; the numerical cube in the HDF5 file is the result.
Requires matplotlib (``pip install matplotlib``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--pixel", type=int, nargs=2, metavar=("ROW", "COL"), default=None,
                        help="pixel to follow in time (default: the hottest pixel of the frame)")
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.h5, "r") as f:
        T = f["temperature_C"]
        ts = f["timestamps"][...] if "timestamps" in f else np.arange(T.shape[0], dtype=float)
        frame = np.asarray(T[args.frame], dtype=float)
        mean_per_frame = np.array([np.nanmean(T[i]) for i in range(T.shape[0])])
        max_per_frame = np.array([np.nanmax(T[i]) for i in range(T.shape[0])])

        if args.pixel is None:
            row, col = np.unravel_index(np.nanargmax(frame), frame.shape)
        else:
            row, col = args.pixel
        series = np.asarray(T[:, row, col], dtype=float)

    # heatmap
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(frame, cmap="inferno")
    ax.set_title(f"Temperature, frame {args.frame} (degC)")
    ax.set_xlabel("column [px]")
    ax.set_ylabel("row [px]")
    fig.colorbar(im, ax=ax, label="degC")
    fig.tight_layout()
    fig.savefig(args.out / f"heatmap_frame_{args.frame:06d}.png", dpi=160)
    plt.close(fig)

    # one pixel over time
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, series, marker=".", lw=1)
    ax.set_title(f"Pixel ({row}, {col}) temperature")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("temperature [degC]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "pixel_timeseries.png", dpi=160)
    plt.close(fig)

    # mean / max over time
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, mean_per_frame, label="mean")
    ax.plot(ts, max_per_frame, label="maximum")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("temperature [degC]")
    ax.set_title("Frame temperature versus time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "mean_temperature.png", dpi=160)
    plt.close(fig)

    print(f"wrote 3 figures to {args.out.resolve()}")
    print(f"pixel ({row}, {col}): {series.min():.2f} … {series.max():.2f} degC")
    print(f"frame mean: {mean_per_frame.min():.2f} … {mean_per_frame.max():.2f} degC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
