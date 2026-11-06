"""Output writer tests: HDF5 structure, streaming, other formats, abort safety."""

from __future__ import annotations

import json

import numpy as np
import pytest

from flir_csq.errors import OutputError
from flir_csq.exporter import Hdf5Exporter, default_output_path, make_exporter

h5py = pytest.importorskip("h5py")


def _fill(exporter, n, h, w, *, save_raw=True):
    exporter.open(n_frames=n, height=h, width=w, save_raw=save_raw, save_mask=False)
    for i in range(n):
        temp = np.full((h, w), 20.0 + i, dtype=np.float32)
        raw = np.full((h, w), 13000 + i, dtype=np.uint16)
        exporter.write(i, temp, raw, i * 0.033, 1_700_000_000_000 + i * 33)
    return exporter.finalize({"conversion": {"mode_effective": "apparent"}, "recording": {},
                              "camera": {}, "timing": {}, "created_utc": "2024-01-01T00:00:00Z",
                              "converter": {"created_utc": "2024-01-01T00:00:00Z"}})


def test_hdf5_layout_dtypes_and_chunks(tmp_path):
    target = tmp_path / "cube.h5"
    path = _fill(Hdf5Exporter(target=target), 6, 12, 16)
    with h5py.File(path, "r") as f:
        assert set(f.keys()) >= {"temperature_C", "raw_counts", "timestamps", "metadata"}
        temp = f["temperature_C"]
        assert temp.shape == (6, 12, 16)
        assert temp.dtype == np.float32
        assert temp.compression == "gzip"
        assert temp.chunks is not None and temp.chunks[0] == 1
        assert temp.attrs["unit"] == "degC"
        assert f["raw_counts"].dtype == np.uint16
        assert f["timestamps"].dtype == np.float64
        assert f["timestamps_unix_ms"].dtype == np.int64
        assert f["temperature_C"][0, 0, 0] == pytest.approx(20.0)
        assert f["temperature_C"][5, 0, 0] == pytest.approx(25.0)
        meta = json.loads(f["metadata"][()])
        assert meta["conversion"]["mode_effective"] == "apparent"


def test_hdf5_rejects_partial_file_after_abort(tmp_path):
    target = tmp_path / "aborted.h5"
    exporter = Hdf5Exporter(target=target)
    exporter.open(n_frames=4, height=4, width=4, save_raw=True, save_mask=False)
    exporter.write(0, np.zeros((4, 4), np.float32), np.zeros((4, 4), np.uint16), 0.0, None)
    exporter.abort()
    assert not target.exists()
    assert not list(tmp_path.glob("*.part"))


def test_npy_export_is_a_memmapped_float32_cube(tmp_path):
    target = tmp_path / "cube.npy"
    path = _fill(make_exporter("npy", target), 3, 5, 7)
    arr = np.load(path, mmap_mode="r")
    assert arr.shape == (3, 5, 7) and arr.dtype == np.float32
    assert arr[2, 0, 0] == pytest.approx(22.0)
    sidecar = json.loads(path.with_name(path.stem + "_metadata.json").read_text())
    assert sidecar["timestamps_s"] == pytest.approx([0.0, 0.033, 0.066])


def test_csv_export_writes_one_numeric_matrix_per_frame(tmp_path):
    target = tmp_path / "frames"
    path = _fill(make_exporter("csv", target), 2, 3, 4)
    files = sorted(path.glob("frame_*.csv"))
    assert [f.name for f in files] == ["frame_000000.csv", "frame_000001.csv"]
    matrix = np.loadtxt(files[1], delimiter=",")
    assert matrix.shape == (3, 4)
    assert matrix[0, 0] == pytest.approx(21.0)


def test_mat_export_roundtrips_through_scipy(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    target = tmp_path / "cube.mat"
    path = _fill(make_exporter("mat", target, mat_version="7"), 2, 4, 4)
    data = scipy_io.loadmat(path)
    # MATLAB orientation: (height, width, frames)
    assert data["temperature_C"].shape == (4, 4, 2)
    assert data["temperature_C"][0, 0, 1] == pytest.approx(21.0, rel=1e-6)
    assert str(data["units"][0]) == "degC"


def test_raw_only_is_refused_for_non_hdf5(tmp_path):
    with pytest.raises(OutputError):
        exporter = make_exporter("npy", tmp_path / "x.npy")
        exporter.open(n_frames=1, height=2, width=2, save_raw=True, save_mask=False,
                      temperature=False)


def test_default_output_path_names():
    from pathlib import Path

    assert default_output_path(Path("/data/experiment_001.csq"), None, "h5") == \
        Path("/data/experiment_001_temperature.h5")
    assert default_output_path(Path("/data/experiment_001.csq"), Path("/tmp/res"), "mat") == \
        Path("/tmp/res/experiment_001_temperature.mat")


def test_unknown_format_is_rejected(tmp_path):
    with pytest.raises(OutputError):
        make_exporter("xlsx", tmp_path / "x.xlsx")
