"""MATLAB v7.3 (HDF5-based) writer with incremental, memory-bounded writes.

A v7.3 ``.mat`` file is an HDF5 file with a 512-byte user block and a 128-byte
MATLAB header, and MATLAB maps its column-major arrays onto HDF5 datasets with
reversed dimensions:

    MATLAB ``A(m, n, p)``  <->  HDF5 dataset of shape ``(p, n, m)``

So a temperature cube that MATLAB should see as ``(height, width, frames)`` is
stored as an HDF5 dataset of shape ``(frames, width, height)``; frame ``k`` is
written at ``dataset[k] = frame.T`` as it is decoded, which is what makes the
export streaming instead of "build the whole cube, then save it".

Each variable carries the attributes MATLAB uses to type it
(``MATLAB_class`` = ``single``/``double``/``uint16``/``char``; ``MATLAB_int_decode``
for character data). Writing was checked against ``hdf5storage``'s v7.3 output:
same 128-byte header, 512-byte user block, version ``0x0200``, ``IM`` endian
marker, and the same dataset orientation and class attributes.

The v5 ("7") format is still used for small cubes because it is readable by
MATLAB, Octave, scipy and everything else; v7.3 is selected automatically when
the cube would approach the 2 GB limit of v7 (:class:`flir_csq.exporter.MatExporter`).
"""

from __future__ import annotations

import datetime as _dt
import platform
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import OutputError

USERBLOCK_SIZE = 512
HEADER_SIZE = 128
VERSION_73 = 0x0200
ENDIAN_MARKER = b"IM"  # little-endian ("IM" == 0x494D read as uint16 LE)

#: MATLAB class attribute for the numpy dtypes this package writes.
MATLAB_CLASS = {
    "float32": "single",
    "float64": "double",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint64": "uint64",
}


def mat73_header(created: _dt.datetime | None = None) -> bytes:
    """The 128-byte MATLAB v7.3 header that goes at the start of the user block."""
    created = created or _dt.datetime.now()
    text = (
        f"MATLAB 7.3 MAT-file, Platform: {platform.system()}-{platform.machine()}, "
        f"Created on: {created:%a %b %d %H:%M:%S %Y} "
        "HDF5 schema 1.00 ."
    )
    payload = text.encode("ascii", errors="replace")[:116].ljust(116, b" ")
    # 116 bytes of text, 8 bytes of subsystem offset (zero), version, endian marker.
    return payload + b"\x00" * 8 + struct.pack("<H", VERSION_73) + ENDIAN_MARKER


def read_header(path: str | Path) -> dict[str, Any]:
    """Inspect a MAT file header without opening HDF5 (used by tests and verification)."""
    with Path(path).open("rb") as fh:
        head = fh.read(HEADER_SIZE)
    if len(head) < HEADER_SIZE:
        return {"valid": False, "reason": "file shorter than a MAT header"}
    version = struct.unpack_from("<H", head, 124)[0]
    endian = head[126:128]
    kind = "v7.3" if version == VERSION_73 else ("v5" if version == 0x0100 else f"0x{version:04x}")
    return {
        "valid": endian in (b"IM", b"MI"),
        "text": head[:116].decode("ascii", errors="replace").rstrip(),
        "version": version,
        "kind": kind,
        "endian": endian.decode("ascii", errors="replace"),
    }


@dataclass
class Mat73Writer:
    """Streaming writer for MATLAB v7.3 files.

    Same call interface as :class:`flir_csq.exporter.Hdf5Exporter`; the conversion
    loop does not know which container it is filling.
    """

    target: Path
    compression: str | None = "gzip"
    compression_level: int = 4
    atomic: bool = True
    _h5: Any = None
    _datasets: dict[str, Any] = field(default_factory=dict)
    _temp: Path | None = None

    # -- lifecycle -----------------------------------------------------------------
    def open(self, *, n_frames: int, height: int, width: int, save_raw: bool,
             save_mask: bool, temperature: bool = True) -> None:
        import h5py

        if not temperature:
            raise OutputError(
                "MATLAB export needs temperatures; raw-only export is HDF5-only",
                hint="Use --format h5 with --raw-only.",
            )
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self._temp = self.target.with_name(self.target.name + ".part") if self.atomic else self.target
        if self._temp.exists():
            self._temp.unlink()
        # 512-byte user block leaves room for the MATLAB header.
        self._h5 = h5py.File(self._temp, "w", userblock_size=USERBLOCK_SIZE)
        with self._temp.open("r+b") as fh:
            fh.write(mat73_header())

        chunk_rows = 1 if n_frames > 4 else max(1, n_frames)
        # HDF5 (frames, width, height) == MATLAB (height, width, frames)
        chunks = (chunk_rows, width, height)
        kwargs: dict[str, Any] = {}
        if self.compression and n_frames > 1:
            kwargs = {"compression": self.compression,
                      "compression_opts": self.compression_level, "shuffle": True}
        self._datasets["temperature_C"] = self._create(
            "temperature_C", (n_frames, width, height), "float32", chunks, kwargs)
        self._datasets["timestamps_s"] = self._create(
            "timestamps_s", (n_frames,), "float64", None, {})
        self._datasets["timestamps_unix_ms"] = self._create(
            "timestamps_unix_ms", (n_frames,), "int64", None, {})
        self._datasets["frame_status"] = self._create(
            "frame_status", (n_frames,), "uint8", None, {})
        if save_raw:
            self._datasets["raw_counts"] = self._create(
                "raw_counts", (n_frames, width, height), "uint16", chunks, kwargs)

    def _create(self, name: str, shape: tuple[int, ...], dtype: str,
                chunks: tuple[int, ...] | None, kwargs: dict[str, Any]):

        ds = self._h5.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks, **kwargs)
        ds.attrs["MATLAB_class"] = np.bytes_(MATLAB_CLASS[dtype])
        return ds

    def write(self, index: int, temperature: np.ndarray, raw: np.ndarray | None,
              timestamp_s: float, timestamp_unix_ms: int | None, status: int = 0) -> None:
        if self._h5 is None:
            raise OutputError("MATLAB writer used before open()")
        self._datasets["frame_status"][index] = int(status)
        # MATLAB (h, w, frames): store frame.T into dataset[index]
        self._datasets["temperature_C"][index] = temperature.T
        self._datasets["timestamps_s"][index] = timestamp_s
        if timestamp_unix_ms is not None:
            self._datasets["timestamps_unix_ms"][index] = timestamp_unix_ms
        else:
            self._datasets["timestamps_unix_ms"][index] = -1
        if raw is not None and "raw_counts" in self._datasets:
            self._datasets["raw_counts"][index] = raw.T

    def finalize(self, metadata: dict[str, Any]) -> Path:
        import json

        if self._h5 is None or self._temp is None:
            raise OutputError("MATLAB writer used before open()")
        payload = json.dumps(metadata, default=str, allow_nan=True)
        # HDF5 (N, 1) -> MATLAB (1, N): a row char vector, which jsondecode() accepts.
        chars = np.frombuffer(payload.encode("utf-16-le"), dtype="<u2").reshape(-1, 1)
        ds = self._h5.create_dataset("metadata_json", data=chars, dtype="uint16")
        ds.attrs["MATLAB_class"] = np.bytes_(b"char")
        ds.attrs["MATLAB_int_decode"] = 2  # UTF-16 code units
        self._h5.flush()
        self._h5.close()
        self._h5 = None
        if self.atomic:
            self._temp.replace(self.target)
        return self.target

    def abort(self) -> None:
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


# ----------------------------------------------------------------------------------
# reading back (verification and tests; no MATLAB required)
# ----------------------------------------------------------------------------------
def read_char(dataset: Any) -> str:
    """Decode a v7.3 char variable (uint16 code units) back to a Python string."""
    codes = np.asarray(dataset).ravel().astype("<u2")
    return str(codes.tobytes().decode("utf-16-le"))


def matlab_shape(hdf5_shape: tuple[int, ...]) -> tuple[int, ...]:
    """HDF5 dataset shape -> the dimensions MATLAB reports."""
    return tuple(reversed(hdf5_shape))


def estimate_v73_bytes(n_frames: int, height: int, width: int, *,
                       save_raw: bool = True, compression: float = 0.7) -> int:
    """Rough on-disk size of the cube (compressed payload plus METADATA)."""
    per_voxel = 4 + (2 if save_raw else 0)
    raw = n_frames * height * width * per_voxel
    return int(raw * compression) + n_frames * 16 + (1 << 20)
