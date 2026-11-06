"""Output writers: HDF5 (default), NumPy, per-frame CSV and MATLAB ``.mat``.

All writers are streaming: frames are appended as they are decoded, so peak
memory is independent of the recording length. HDF5 is the default because a
464x348x(frames) float32 cube is ~6.2 MB per 10 frames and recordings are
routinely thousands of frames long.

HDF5 layout written by :class:`Hdf5Exporter`::

    /temperature_C        float32 (frames, height, width)  degC, NaN = invalid
    /raw_counts           uint16  (frames, height, width)  as decoded, optional
    /timestamps           float64 (frames,)                s since first frame
    /timestamps_unix_ms   int64   (frames,)                camera clock, epoch ms
    /valid_mask           uint8   (frames, height, width)  optional
    /metadata             JSON string (scalar)             everything else

Each dataset carries its own ``unit`` / ``source`` attributes; the JSON metadata
records the decoder, the calibration constants, the conversion mode actually
applied, per-frame statistics and every warning raised.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .errors import OutputError

CONVERTER_VERSION = "0.9.1"


class Exporter(Protocol):  # pragma: no cover - structural typing only
    def open(self, *, n_frames: int, height: int, width: int, save_raw: bool,
             save_mask: bool, temperature: bool = True) -> None: ...
    def write(self, index: int, temperature: np.ndarray | None, raw: np.ndarray | None,
              timestamp_s: float, timestamp_unix_ms: int | None, status: int = 0) -> None: ...
    def finalize(self, metadata: dict[str, Any]) -> Path: ...
    def abort(self) -> None: ...


def _temp_path(target: Path) -> Path:
    return target.with_name(target.name + ".part")


@dataclass
class Hdf5Exporter:
    """Streaming HDF5 writer with chunking and compression."""

    target: Path
    compression: str | None = "gzip"
    compression_level: int = 4
    atomic: bool = True
    _h5: Any = None
    _datasets: dict[str, Any] = field(default_factory=dict)
    _temp: Path | None = None
    _closed: bool = False

    def open(self, *, n_frames: int, height: int, width: int, save_raw: bool,
             save_mask: bool, temperature: bool = True) -> None:
        import h5py

        self.target.parent.mkdir(parents=True, exist_ok=True)
        self._temp = _temp_path(self.target) if self.atomic else self.target
        if self._temp.exists():
            self._temp.unlink()
        chunk_rows = 1 if n_frames > 4 else max(1, n_frames)
        chunks = (chunk_rows, min(height, 512), min(width, 512))
        kwargs: dict[str, Any] = {}
        if self.compression and n_frames > 1:
            kwargs = {"compression": self.compression, "compression_opts": self.compression_level,
                      "shuffle": True}
        self._h5 = h5py.File(self._temp, "w")
        if temperature:
            self._datasets["temperature_C"] = self._h5.create_dataset(
                "temperature_C", shape=(n_frames, height, width), dtype="float32",
                chunks=chunks, **kwargs)
        self._datasets["timestamps"] = self._h5.create_dataset(
            "timestamps", shape=(n_frames,), dtype="float64")
        self._datasets["timestamps_unix_ms"] = self._h5.create_dataset(
            "timestamps_unix_ms", shape=(n_frames,), dtype="int64", fillvalue=-1)
        self._datasets["frame_status"] = self._h5.create_dataset(
            "frame_status", shape=(n_frames,), dtype="uint8", fillvalue=0)
        if save_raw:
            self._datasets["raw_counts"] = self._h5.create_dataset(
                "raw_counts", shape=(n_frames, height, width), dtype="uint16",
                chunks=chunks, **kwargs)
        if save_mask:
            mask_chunks = (chunk_rows, height, width)
            self._datasets["valid_mask"] = self._h5.create_dataset(
                "valid_mask", shape=(n_frames, height, width), dtype="uint8",
                chunks=mask_chunks, compression="gzip", compression_opts=1)

    def write(self, index: int, temperature: np.ndarray | None, raw: np.ndarray | None,
              timestamp_s: float, timestamp_unix_ms: int | None, status: int = 0) -> None:
        if self._h5 is None:
            raise OutputError("HDF5 exporter used before open()")
        self._datasets["frame_status"][index] = int(status)
        if temperature is not None and "temperature_C" in self._datasets:
            self._datasets["temperature_C"][index] = temperature
        self._datasets["timestamps"][index] = timestamp_s
        if timestamp_unix_ms is not None:
            self._datasets["timestamps_unix_ms"][index] = timestamp_unix_ms
        if raw is not None and "raw_counts" in self._datasets:
            self._datasets["raw_counts"][index] = raw
        if "valid_mask" in self._datasets and temperature is not None:
            self._datasets["valid_mask"][index] = np.isfinite(temperature).astype(np.uint8)

    def finalize(self, metadata: dict[str, Any]) -> Path:
        if self._h5 is None or self._temp is None:
            raise OutputError("HDF5 exporter used before open()")
        import h5py

        h5 = self._h5
        attrs = {
            "temperature_C": {"unit": "degC", "quantity": "calibrated object temperature",
                              "invalid_value": "NaN"},
            "raw_counts": {"unit": "sensor counts", "note":
                           "as decoded from the recording; not temperatures"},
            "timestamps": {"unit": "s", "reference": "seconds since the first converted frame",
                           "source": metadata.get("timing", {}).get("timestamp_source", "unknown")},
            "timestamps_unix_ms": {"unit": "ms", "reference": "POSIX epoch, camera clock (UTC)",
                                   "fillvalue": -1},
            "frame_status": {"meaning": "0 = decoded, 1 = decode failed (temperature NaN, "
                                        "timestamp preserved)"},
        }
        for name, ds in self._datasets.items():
            for key, value in attrs.get(name, {}).items():
                ds.attrs[key] = value
            ds.attrs["converter"] = f"flir-csq {CONVERTER_VERSION}"
        h5.attrs["converter"] = f"flir-csq {CONVERTER_VERSION}"
        h5.attrs["source_file"] = str(metadata.get("recording", {}).get("path", ""))
        h5.attrs["camera_model"] = str(metadata.get("camera", {}).get("camera_model") or "unknown")
        h5.attrs["conversion_mode"] = str(metadata.get("conversion", {}).get("mode_effective"))
        h5.attrs["units"] = "degC"
        h5.attrs["created_utc"] = str(metadata.get("converter", {}).get("created_utc", ""))
        payload = json.dumps(metadata, default=_json_default, allow_nan=True)
        h5.create_dataset("metadata", data=payload, dtype=h5py.string_dtype("utf-8"))
        h5.flush()
        h5.close()
        self._h5 = None
        self.target.parent.mkdir(parents=True, exist_ok=True)
        if self.atomic:
            os.replace(self._temp, self.target)
        self._closed = True
        return self.target

    def abort(self) -> None:
        """Close and delete the partial file - a cancelled run leaves no half cube."""
        try:
            if self._h5 is not None:
                self._h5.close()
        finally:
            self._h5 = None
            if self._temp is not None and self._temp.exists():
                try:
                    self._temp.unlink()
                except OSError:
                    pass


@dataclass
class NpyExporter:
    """``.npy`` written through a memory map, so it streams too."""

    target: Path
    _array: Any = None
    _timestamps: list[float] = field(default_factory=list)

    def open(self, *, n_frames: int, height: int, width: int, save_raw: bool,
             save_mask: bool, temperature: bool = True) -> None:
        from numpy.lib.format import open_memmap

        _require_temperature(temperature, "NumPy")
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self._array = open_memmap(self.target, mode="w+", dtype="float32",
                                  shape=(n_frames, height, width))

    def write(self, index: int, temperature: np.ndarray | None, raw: np.ndarray | None,
              timestamp_s: float, timestamp_unix_ms: int | None, status: int = 0) -> None:
        self._array[index] = temperature
        self._timestamps.append(float(timestamp_s))

    def finalize(self, metadata: dict[str, Any]) -> Path:
        self._array.flush()
        del self._array
        self._array = None
        sidecar = self.target.with_name(self.target.stem + "_metadata.json")
        sidecar.write_text(json.dumps({**metadata, "timestamps_s": self._timestamps},
                                      default=_json_default, indent=2), encoding="utf-8")
        return self.target

    def abort(self) -> None:
        self._array = None
        if self.target.exists():
            try:
                self.target.unlink()
            except OSError:
                pass


@dataclass
class CsvExporter:
    """One numeric CSV per frame - convenient for MATLAB/Excel spot checks."""

    target: Path
    _folder: Path | None = None
    _decimals: int = 4

    def open(self, *, n_frames: int, height: int, width: int, save_raw: bool,
             save_mask: bool, temperature: bool = True) -> None:
        _require_temperature(temperature, "CSV")
        self._folder = self.target
        self._folder.mkdir(parents=True, exist_ok=True)

    def write(self, index: int, temperature: np.ndarray | None, raw: np.ndarray | None,
              timestamp_s: float, timestamp_unix_ms: int | None, status: int = 0) -> None:
        assert self._folder is not None
        path = self._folder / f"frame_{index:06d}.csv"
        fmt = f"%.{self._decimals}f"
        np.savetxt(path, temperature, delimiter=",", fmt=fmt, encoding="utf-8")

    def finalize(self, metadata: dict[str, Any]) -> Path:
        assert self._folder is not None
        (self._folder / "metadata.json").write_text(
            json.dumps(metadata, default=_json_default, indent=2), encoding="utf-8")
        return self._folder

    def abort(self) -> None:
        if self._folder is not None and self._folder.exists():
            import shutil

            shutil.rmtree(self._folder, ignore_errors=True)


@dataclass
class MatExporter:
    """MATLAB ``.mat`` cube, v7.3 (HDF5) or v5 depending on size.

    MATLAB sees ``temperature_C`` as ``(height, width, frames)`` in both formats.

    * **v7.3** (MATLAB >= 7.3, "``-v7.3``"): an HDF5 container with a 512-byte user
      block. Frames are written one at a time while they are decoded, so a cube
      far larger than 2 GB costs one frame of RAM (see
      :class:`flir_csq.mat73.Mat73Writer`).
    * **v5/v7** (``scipy.io.savemat``): universal (MATLAB, Octave, scipy) but the
      format itself cannot exceed 2 GB and ``savemat`` needs the cube in RAM. It
      is used for small cubes only, and refused above the size limit.

    ``version="auto"`` picks v7.3 when the estimated size approaches the v7 limit.
    """

    target: Path
    compressed: bool = True
    version: str = "auto"  # "auto" | "7" | "7.3"
    threshold_bytes: int = 256_000_000
    compression_level: int = 4
    atomic: bool = True
    used_version: str = ""
    _impl: Any = None
    _array: Any = None
    _timestamps: list[float] = field(default_factory=list)
    _unix_ms: list[int] = field(default_factory=list)
    _status: list[int] = field(default_factory=list)
    _scratch: Path | None = None
    _n_frames: int = 0

    def open(self, *, n_frames: int, height: int, width: int, save_raw: bool,
             save_mask: bool, temperature: bool = True) -> None:
        from numpy.lib.format import open_memmap

        from .mat73 import Mat73Writer, estimate_v73_bytes

        _require_temperature(temperature, "MATLAB")
        self._n_frames = n_frames
        estimated = estimate_v73_bytes(n_frames, height, width, save_raw=save_raw)
        if self.version == "auto":
            use_73 = estimated > self.threshold_bytes
        elif self.version in ("7.3", "v7.3", "73"):
            use_73 = True
        elif self.version in ("7", "v7", "5"):
            use_73 = False
            # v7 cannot exceed 2 GB; the configured threshold keeps a safety margin.
            limit = min(2_000_000_000, self.threshold_bytes) if self.threshold_bytes > 0 \
                else 2_000_000_000
            if estimated > limit:
                raise OutputError(
                    f"this cube is estimated at {estimated / 1e9:.2f} GB, above the "
                    f"{limit / 1e9:.2f} GB limit usable with the MATLAB v7 format",
                    hint="Use --mat-version 7.3 (or auto) to write a MATLAB v7.3 file, "
                         "which has no such limit.",
                )
        else:
            raise OutputError(f"unknown MATLAB format {self.version!r}",
                              hint="choose auto, 7 or 7.3")

        if not use_73 and self.version == "auto":
            # scipy is only needed for the small-cube v7 container; without it, stream to 7.3.
            try:
                import scipy.io  # noqa: F401
            except ImportError:
                import logging

                logging.getLogger("flir_csq").info(
                    "scipy is not installed: writing MATLAB v7.3 instead of v7")
                use_73 = True

        self.used_version = "7.3" if use_73 else "7"
        if use_73:
            self._impl = Mat73Writer(target=self.target, compression_level=self.compression_level,
                                     atomic=self.atomic)
            self._impl.open(n_frames=n_frames, height=height, width=width, save_raw=save_raw,
                            save_mask=save_mask, temperature=temperature)
            return

        self.target.parent.mkdir(parents=True, exist_ok=True)
        self._scratch = self.target.with_name(self.target.name + ".scratch.npy")
        self._array = open_memmap(self._scratch, mode="w+", dtype="float32",
                                  shape=(n_frames, height, width))

    def write(self, index: int, temperature: np.ndarray | None, raw: np.ndarray | None,
              timestamp_s: float, timestamp_unix_ms: int | None, status: int = 0) -> None:
        if self._impl is not None:
            self._impl.write(index, temperature, raw, timestamp_s, timestamp_unix_ms,
                             status=status)
            return
        assert self._array is not None and temperature is not None
        self._array[index] = temperature
        self._timestamps.append(float(timestamp_s))
        self._unix_ms.append(int(timestamp_unix_ms) if timestamp_unix_ms is not None else -1)
        self._status.append(int(status))

    def finalize(self, metadata: dict[str, Any]) -> Path:
        if self._impl is not None:
            produced = self._impl.finalize(metadata)
            self._impl = None
            return produced
        try:
            from scipy.io import savemat
        except ImportError as exc:  # pragma: no cover - depends on install
            raise OutputError(
                "MATLAB v7 export needs scipy",
                hint="pip install scipy, or use --mat-version 7.3 (HDF5, no scipy needed)",
            ) from exc
        assert self._array is not None
        self._array.flush()
        cube = np.asarray(self._array)                     # (frames, h, w)
        hwf = np.transpose(cube, (1, 2, 0))                # (h, w, frames), as MATLAB wants
        savemat(self.target, {
            "temperature_C": hwf,
            "timestamps_s": np.asarray(self._timestamps, dtype=np.float64).reshape(-1, 1),
            "timestamps_unix_ms": np.asarray(self._unix_ms, dtype=np.int64).reshape(-1, 1),
            "frame_status": np.asarray(self._status, dtype=np.uint8).reshape(-1, 1),
            "metadata_json": json.dumps(metadata, default=_json_default),
            "units": "degC",
        }, do_compression=self.compressed, oned_as="column")
        del hwf, cube
        self._array = None
        if self._scratch and self._scratch.exists():
            self._scratch.unlink()
        return self.target

    def abort(self) -> None:
        if self._impl is not None:
            self._impl.abort()
            self._impl = None
        self._array = None
        for path in (self.target, self._scratch):
            if path and path.exists():
                try:
                    path.unlink()
                except OSError:  # noqa: BLE001 - best effort cleanup
                    pass


def _require_temperature(temperature: bool, what: str) -> None:
    if not temperature:
        raise OutputError(
            f"raw-only export is only supported for HDF5, not {what}",
            hint="Use --format h5 when the recording has no usable calibration.",
        )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def make_exporter(fmt: str, target: Path, *, compression_level: int = 4,
                  mat_compressed: bool = True, atomic: bool = True,
                  mat_version: str = "auto",
                  mat73_threshold_bytes: int = 256_000_000) -> Exporter:
    fmt = fmt.lower()
    if fmt in ("h5", "hdf5"):
        return Hdf5Exporter(target=target, compression_level=compression_level, atomic=atomic)
    if fmt in ("npy", "numpy"):
        return NpyExporter(target=target)
    if fmt == "csv":
        return CsvExporter(target=target)
    if fmt in ("mat", "matlab"):
        return MatExporter(target=target, compressed=mat_compressed, version=mat_version,
                           threshold_bytes=mat73_threshold_bytes,
                           compression_level=compression_level, atomic=atomic)
    raise OutputError(f"unknown output format {fmt!r}", hint="choose h5, npy, csv or mat")


#: Filename suffix used for each format, given the input stem.
SUFFIXES = {"h5": "_temperature.h5", "npy": "_temperature.npy", "csv": "_temperature_csv",
            "mat": "_temperature.mat"}


def default_output_path(input_path: Path, output_dir: Path | None, fmt: str) -> Path:
    stem = input_path.stem
    directory = output_dir if output_dir is not None else input_path.parent
    return Path(directory) / f"{stem}{SUFFIXES[fmt]}"


def rewrite_hdf5_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    """Replace the ``/metadata`` JSON block after verification has been recorded.

    The block is written when the cube is closed; verification and per-file
    sizes only exist afterwards, so they are merged in with this second pass
    (an attribute-sized rewrite, not a rewrite of the data).
    """
    import h5py

    with h5py.File(path, "a") as f:
        payload = json.dumps(metadata, default=_json_default, allow_nan=True)
        if "metadata" in f:
            del f["metadata"]
        f.create_dataset("metadata", data=payload, dtype=h5py.string_dtype("utf-8"))
        f.attrs["conversion_mode"] = str(metadata.get("conversion", {}).get("mode_effective"))
        f.attrs["frames_converted"] = int(metadata.get("recording", {}).get("n_frames_converted", 0))
