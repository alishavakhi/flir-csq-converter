"""MATLAB export tests: v7 orientation, v7.3 layout, streaming, batch and CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from flir_csq.batch import BatchOptions, run_batch
from flir_csq.converter import ConversionOptions, convert
from flir_csq.errors import OutputError
from flir_csq.exporter import MatExporter
from flir_csq.mat73 import (
    HEADER_SIZE,
    USERBLOCK_SIZE,
    estimate_v73_bytes,
    matlab_shape,
    read_char,
    read_header,
)
from flir_csq.verify import verify_mat

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "convert_csq.py"
h5py = pytest.importorskip("h5py")


def _convert(recording, out_dir, **kwargs):
    kwargs.setdefault("formats", ("mat",))
    options = ConversionOptions(input_path=recording["path"], output_dir=out_dir, **kwargs)
    return convert(options)


# -- v7 (scipy) --------------------------------------------------------------------
def test_v7_mat_has_matlab_orientation_and_values(jpegls_recording, tmp_path):
    from scipy.io import loadmat

    result = _convert(jpegls_recording, tmp_path, mat_version="7")
    path = result.outputs["mat"]
    assert path.suffix == ".mat"
    data = loadmat(path)
    cube = data["temperature_C"]
    frames, h, w = jpegls_recording["temperatures_c"].shape
    assert cube.shape == (h, w, frames), "MATLAB must see (height, width, frames)"
    assert cube.dtype == np.float32
    truth = np.transpose(jpegls_recording["temperatures_c"], (1, 2, 0)).astype(np.float32)
    np.testing.assert_allclose(cube, truth, atol=0.02)
    assert data["timestamps_s"].shape == (frames, 1)
    stamps = np.array(jpegls_recording["timestamps_ms"], dtype=float)
    np.testing.assert_allclose(data["timestamps_s"].ravel(),
                               (stamps - stamps[0]) / 1000.0, atol=1e-6)
    meta = json.loads(str(data["metadata_json"].ravel()[0]))
    assert meta["conversion"]["units"] == "degC"
    assert str(data["units"].ravel()[0]) == "degC"


def test_v7_is_refused_above_the_format_limit(tmp_path):
    exporter = MatExporter(target=tmp_path / "big.mat", version="7", threshold_bytes=10)
    with pytest.raises(OutputError) as exc:
        exporter.open(n_frames=2, height=4, width=4, save_raw=True, save_mask=False)
    assert "2 GB" in str(exc.value) or "v7" in str(exc.value)
    assert not list(tmp_path.glob("*.mat"))


# -- v7.3 (HDF5) -------------------------------------------------------------------
def test_v73_header_and_userblock(tmp_path):
    exporter = MatExporter(target=tmp_path / "cube.mat", version="7.3")
    exporter.open(n_frames=3, height=4, width=5, save_raw=True, save_mask=False)
    for i in range(3):
        exporter.write(i, np.full((4, 5), 20.0 + i, np.float32),
                       np.full((4, 5), 13000 + i, np.uint16), i * 0.033, 1_700_000_000_000 + i * 33)
    path = exporter.finalize({"conversion": {"units": "degC"}})

    header = read_header(path)
    assert header["kind"] == "v7.3" and header["valid"]
    assert header["endian"] == "IM"
    assert path.read_bytes()[:19] == b"MATLAB 7.3 MAT-file"
    with h5py.File(path, "r") as f:
        assert f.userblock_size == USERBLOCK_SIZE
        assert dict(f.attrs) == {}  # MATLAB does not require root attributes
    assert path.stat().st_size > USERBLOCK_SIZE + HEADER_SIZE


def test_v73_orientation_attributes_and_metadata(tmp_path):
    exporter = MatExporter(target=tmp_path / "cube.mat", version="7.3")
    exporter.open(n_frames=4, height=6, width=7, save_raw=True, save_mask=False)
    for i in range(4):
        exporter.write(i, np.full((6, 7), 20.0 + i, np.float32),
                       np.full((6, 7), 13000 + i, np.uint16), i * 0.5,
                       1_700_000_000_000 + i * 500)
    metadata = {"conversion": {"units": "degC", "mode_effective": "object"},
                "camera": {"camera_model": "FLIR T540"}}
    path = exporter.finalize(metadata)

    with h5py.File(path, "r") as f:
        ds = f["temperature_C"]
        assert ds.shape == (4, 7, 6), "HDF5 is (frames, width, height)"
        assert matlab_shape(ds.shape) == (6, 7, 4), "MATLAB is (height, width, frames)"
        assert ds.dtype == np.float32
        assert ds.chunks[1:] == (7, 6)  # one chunk spans a frame's plane
        assert ds.compression == "gzip"
        assert ds.attrs["MATLAB_class"] == b"single"
        assert f["raw_counts"].shape == (4, 7, 6)
        assert f["raw_counts"].attrs["MATLAB_class"] == b"uint16"
        assert f["timestamps_s"].shape == (4,)
        assert f["timestamps_s"].attrs["MATLAB_class"] == b"double"
        assert f["metadata_json"].attrs["MATLAB_class"] == b"char"

        # frame k of MATLAB is dataset[k].T
        np.testing.assert_allclose(np.asarray(ds[2]).T, np.full((6, 7), 22.0, np.float32))
        meta = json.loads(read_char(f["metadata_json"]))
        assert meta["conversion"]["mode_effective"] == "object"
        assert meta["camera"]["camera_model"] == "FLIR T540"


def test_v73_without_raw_counts_is_smaller(tmp_path):
    sizes = {}
    for save_raw in (True, False):
        exporter = MatExporter(target=tmp_path / f"cube_{save_raw}.mat", version="7.3")
        exporter.open(n_frames=5, height=8, width=9, save_raw=save_raw, save_mask=False)
        for i in range(5):
            exporter.write(i, np.full((8, 9), 20.0, np.float32), np.full((8, 9), 13000, np.uint16),
                           0.0, None)
        path = exporter.finalize({"conversion": {"units": "degC"}})
        with h5py.File(path) as f:
            assert ("raw_counts" in f) is save_raw
        sizes[save_raw] = path.stat().st_size
    assert sizes[False] < sizes[True]


def test_auto_selects_v73_for_large_cubes(tmp_path):
    small = MatExporter(target=tmp_path / "small.mat", version="auto",
                        threshold_bytes=10_000_000)
    small.open(n_frames=2, height=4, width=4, save_raw=False, save_mask=False)
    assert small.used_version == "7"
    small.abort()

    large = MatExporter(target=tmp_path / "large.mat", version="auto", threshold_bytes=1000)
    large.open(n_frames=2, height=4, width=4, save_raw=True, save_mask=False)
    assert large.used_version == "7.3"
    large.abort()


def test_estimate_scales_with_frames():
    a = estimate_v73_bytes(100, 348, 464, save_raw=True)
    b = estimate_v73_bytes(1000, 348, 464, save_raw=True)
    assert b > 5 * a
    assert 3e9 < estimate_v73_bytes(10_000, 348, 464, save_raw=True) < 12e9


def test_abort_leaves_no_mat_file(tmp_path):
    exporter = MatExporter(target=tmp_path / "aborted.mat", version="7.3")
    exporter.open(n_frames=3, height=4, width=4, save_raw=True, save_mask=False)
    exporter.write(0, np.zeros((4, 4), np.float32), np.zeros((4, 4), np.uint16), 0.0, None)
    exporter.abort()
    assert not list(tmp_path.glob("*.mat"))
    assert not list(tmp_path.glob("*.part"))


def test_two_independent_readers_agree_with_each_other(jpegls_recording, tmp_path):
    """mat73 and hdf5storage are independent v7.3 readers; both must see (h, w, frames)."""
    mat73 = pytest.importorskip("mat73")
    hdf5storage = pytest.importorskip("hdf5storage")

    result = _convert(jpegls_recording, tmp_path, mat_version="7.3")
    path = result.outputs["mat"]

    a = mat73.loadmat(str(path))
    b = hdf5storage.loadmat(str(path))
    frames, h, w = jpegls_recording["temperatures_c"].shape
    assert np.shape(a["temperature_C"]) == (h, w, frames)
    assert np.shape(b["temperature_C"]) == (h, w, frames)
    np.testing.assert_allclose(a["temperature_C"], b["temperature_C"], atol=1e-6)
    assert isinstance(a["metadata_json"], str)
    assert json.loads(a["metadata_json"])["conversion"]["units"] == "degC"


# -- verification ------------------------------------------------------------------
def test_verify_mat_accepts_both_containers(jpegls_recording, tmp_path):
    for version in ("7", "7.3"):
        result = _convert(jpegls_recording, tmp_path / f"v{version}", mat_version=version)
        report = verify_mat(result.outputs["mat"])
        assert report["passed"], report["problems"]
        assert report["shape"] == [48, 64, 8]


def test_verify_mat_reports_a_damaged_cube(jpegls_recording, tmp_path):
    result = _convert(jpegls_recording, tmp_path, mat_version="7.3")
    path = result.outputs["mat"]
    with h5py.File(path, "a") as f:
        del f["timestamps_s"]
    report = verify_mat(path)
    assert report["passed"] is False
    assert any("timestamps_s" in p for p in report["problems"])


def test_verify_mat_reports_a_missing_file(tmp_path):
    report = verify_mat(tmp_path / "nope.mat")
    assert report["passed"] is False


# -- batch and CLI -----------------------------------------------------------------
def test_batch_writes_mat_per_recording(tmp_path):
    from flir_csq.synthetic import make_synthetic_csq

    inp = tmp_path / "input"
    (inp / "campaign").mkdir(parents=True)
    make_synthetic_csq(inp / "exp_001.csq", n_frames=5, height=16, width=20)
    make_synthetic_csq(inp / "campaign" / "exp_002.csq", n_frames=3, height=16, width=20)

    result = run_batch(BatchOptions(input_dir=inp, output_dir=tmp_path / "out",
                                    formats=("mat",), mat_version="7.3"))
    assert result.converted == 2
    files = sorted((tmp_path / "out").rglob("*.mat"))
    assert {f.name for f in files} == {"exp_001_temperature.mat", "exp_002_temperature.mat"}
    with h5py.File(tmp_path / "out" / "campaign" / "exp_002_temperature.mat") as f:
        assert matlab_shape(f["temperature_C"].shape) == (16, 20, 3)
        assert f["metadata_json"].attrs["MATLAB_class"] == b"char"


def test_cli_mat_output(jpegls_recording, tmp_path):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(jpegls_recording["path"]),
         "--output", str(tmp_path), "--format", "mat", "--mat-version", "7.3"],
        capture_output=True, text=True, check=False, timeout=300)
    assert proc.returncode == 0, proc.stderr
    path = tmp_path / "jpegls_t540_temperature.mat"
    assert path.is_file()
    assert "verify     PASS" in proc.stdout
    verify = subprocess.run([sys.executable, str(SCRIPT), "--verify", str(path)],
                            capture_output=True, text=True, check=False, timeout=120)
    assert verify.returncode == 0 and verify.stdout.startswith("PASS")


def test_cli_batch_mat(tmp_path):
    from flir_csq.synthetic import make_synthetic_csq

    inp = tmp_path / "in"
    inp.mkdir()
    make_synthetic_csq(inp / "a.csq", n_frames=3, height=10, width=12)
    make_synthetic_csq(inp / "b.csq", n_frames=2, height=10, width=12)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--input-dir", str(inp),
         "--output-dir", str(tmp_path / "out"), "--format", "mat", "--mat-version", "7.3"],
        capture_output=True, text=True, check=False, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "out" / "a_temperature.mat").is_file()
    assert (tmp_path / "out" / "b_temperature.mat").is_file()
    assert "2 converted" in proc.stdout


@pytest.mark.skipif(sys.platform != "linux", reason="the wrapper targets Linux servers")
def test_run_server_with_mat_format(tmp_path):
    from flir_csq.synthetic import make_synthetic_csq

    inp = tmp_path / "input"
    inp.mkdir()
    make_synthetic_csq(inp / "rec.csq", n_frames=3, height=10, width=12)
    out = tmp_path / "output"
    env = dict(os.environ, FLIR_PYTHON=sys.executable, FLIR_FORMATS="mat",
               FLIR_MAT_VERSION="7.3")
    env.pop("FLIR_VENV", None)
    proc = subprocess.run(["bash", str(ROOT / "run_server.sh"), str(inp), str(out)],
                          capture_output=True, text=True, env=env, timeout=300, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "rec_temperature.mat").is_file()


# -- memory ------------------------------------------------------------------------
MEM_SCRIPT = textwrap.dedent(
    """
    import json, resource, sys
    sys.path.insert(0, {src!r})
    from flir_csq.converter import ConversionOptions, convert
    out = convert(ConversionOptions(input_path={path!r}, output_dir={out!r},
                                    formats=("mat",), mat_version="7.3",
                                    save_raw=False, save_metadata=False))
    print(json.dumps({{"frames": out.frames_written,
                       "peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}}))
    """
)


def _peak_mb(frames: int, tmp_path: Path) -> float:
    from flir_csq.synthetic import make_synthetic_csq

    path = tmp_path / f"long_{frames}.csq"
    make_synthetic_csq(path, n_frames=frames, height=48, width=64)
    script = MEM_SCRIPT.format(src=str(ROOT / "src"), path=str(path),
                               out=str(tmp_path / f"out_{frames}"))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=600, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])["peak_mb"]


def test_v73_writing_streams_instead_of_loading_the_cube(tmp_path):
    small = _peak_mb(30, tmp_path)
    large = _peak_mb(400, tmp_path)
    print(f"peak RSS: 30 frames {small:.1f} MB, 400 frames {large:.1f} MB")
    assert large - small < 60.0, (
        f"peak memory grew by {large - small:.1f} MB between 30 and 400 frames - "
        "the MAT writer is not streaming"
    )
