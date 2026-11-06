#!/usr/bin/env python3
"""Load a temperature cube and print its numbers - no plotting, no dependencies beyond HDF5.

    python examples/read_hdf5.py experiment_001_temperature.h5

This is the check that matters: the file contains real temperatures in degC for
every pixel of every frame, not colours.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path)
    parser.add_argument("--metadata", action="store_true", help="print the full metadata block")
    args = parser.parse_args(argv)

    with h5py.File(args.h5, "r") as f:
        T = f["temperature_C"]
        print("Shape:", T.shape)
        print("Dtype:", T.dtype, "| units:", T.attrs.get("unit"))
        print("First frame:")
        print(T[0])

        frame = T[0]
        print("Minimum temperature:", np.nanmin(frame))
        print("Maximum temperature:", np.nanmax(frame))
        print("Mean temperature:", np.nanmean(frame))
        print("Center pixel:", frame[frame.shape[0] // 2, frame.shape[1] // 2])

        if "timestamps" in f:
            ts = f["timestamps"][...]
            print(f"Timestamps: {ts[0]:.3f} … {ts[-1]:.3f} s "
                  f"({len(ts)} frames, source: {f['timestamps'].attrs.get('source')})")
        if "raw_counts" in f:
            raw = f["raw_counts"][0]
            print(f"Raw counts of frame 0: min {raw.min()}, max {raw.max()} "
                  "(sensor counts, not temperatures)")
        if args.metadata:
            meta = json.loads(f["metadata"][()])
            print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
