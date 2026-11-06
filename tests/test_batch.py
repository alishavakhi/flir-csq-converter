"""Batch (headless) conversion tests: discovery, skip/resume, error isolation, logs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from flir_csq.batch import (
    BatchOptions,
    discover_recordings,
    load_state,
    run_batch,
)
from flir_csq.synthetic import SyntheticCamera, make_field, make_synthetic_csq

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "convert_csq.py"
h5py = pytest.importorskip("h5py")


def _make(root: Path, name: str, *, frames: int = 4, calibrated: bool = True,
          compression: str = "jpegls") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    camera = SyntheticCamera(with_planck=calibrated, width=20, height=16)
    make_synthetic_csq(path, n_frames=frames, height=16, width=20,
                       compression=compression, camera=camera,
                       temperatures=make_field(frames, 16, 20))
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> dict:
    """A realistic server upload: good files, a nested file, a corrupt one."""
    inp = tmp_path / "input"
    good = [_make(inp, "exp_001.csq", frames=5), _make(inp, "exp_002.csq", frames=3)]
    nested = _make(inp / "campaign_b", "exp_003.csq", frames=2)
    corrupt = inp / "exp_broken.csq"
    corrupt.write_bytes(b"not a FLIR recording" * 64)
    (inp / "README.txt").write_text("notes, not a recording")
    return {"input": inp, "out": tmp_path / "output",
            "good": good, "nested": nested, "corrupt": corrupt}


# -- discovery ---------------------------------------------------------------------
def test_discovery_is_recursive_and_sorted(dataset):
    found = discover_recordings(dataset["input"])
    assert {p.name for p in found} == {"exp_001.csq", "exp_002.csq", "exp_broken.csq",
                                       "exp_003.csq"}
    assert dataset["nested"] in found
    assert all(p.suffix.lower() == ".csq" for p in found)
    assert dataset["input"] / "README.txt" not in found
    assert found.index(dataset["input"] / "exp_001.csq") < \
        found.index(dataset["input"] / "exp_002.csq"), "order must be stable"


def test_discovery_can_be_non_recursive(dataset):
    found = discover_recordings(dataset["input"], recursive=False)
    assert dataset["nested"] not in found
    assert dataset["good"][0] in found


def test_discovery_is_case_insensitive(tmp_path):
    _make(tmp_path, "UPPER.CSQ")
    _make(tmp_path, "lower.csq")
    found = discover_recordings(tmp_path)
    assert {p.name for p in found} == {"UPPER.CSQ", "lower.csq"}


# -- conversion --------------------------------------------------------------------
def test_batch_converts_every_recording_and_keeps_names(dataset):
    result = run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                                    formats=("h5",)))
    assert result.converted == 3
    assert result.failed == 1
    assert result.exit_code == 4
    for name in ("exp_001_temperature.h5", "exp_002_temperature.h5"):
        assert (dataset["out"] / name).is_file()
    assert (dataset["out"] / "campaign_b" / "exp_003_temperature.h5").is_file()
    with h5py.File(dataset["out"] / "exp_001_temperature.h5") as f:
        assert f["temperature_C"].shape == (5, 16, 20)
        assert f["temperature_C"].dtype.name == "float32"
        assert f["temperature_C"].attrs["unit"] == "degC"
        assert f["timestamps"].shape == (5,)
        assert "metadata" in f


def test_corrupt_file_does_not_stop_the_others(dataset):
    result = run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                                    formats=("h5",)))
    failed = [o for o in result.outcomes if o.status == "failed"]
    assert len(failed) == 1 and failed[0].input.name == "exp_broken.csq"
    assert "FFF" in (failed[0].error or "")
    assert result.converted == 3, "a corrupt file must not lose the successful conversions"


def test_uncalibrated_recording_fails_with_a_hint_and_does_not_block(dataset):
    _make(dataset["input"], "no_planck.csq", frames=2, calibrated=False)
    result = run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                                    formats=("h5",)))
    failure = next(o for o in result.outcomes if o.input.name == "no_planck.csq")
    assert failure.status == "failed"
    assert "Planck" in (failure.error or "")
    assert "--raw-only" in (failure.hint or "")
    assert result.converted == 3


def test_on_error_stop_halts_after_the_first_failure(tmp_path):
    inp = tmp_path / "in"
    _make(inp, "a_good.csq")
    (inp / "b_bad.csq").write_bytes(b"junk" * 100)
    _make(inp, "c_good.csq")
    result = run_batch(BatchOptions(input_dir=inp, output_dir=tmp_path / "out",
                                    formats=("h5",), continue_on_error=False))
    assert result.converted == 1
    assert result.failed == 1
    assert len(result.outcomes) == 2, "processing must stop after the failure"


def test_limit_processes_only_the_first_n(dataset):
    result = run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                                    formats=("h5",), limit=1))
    assert len(result.outcomes) == 1


def test_batch_can_write_several_formats_at_once(dataset):
    result = run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                                    formats=("h5", "csv")))
    assert result.converted == 3
    assert (dataset["out"] / "exp_001_temperature.h5").is_file()
    csv_dir = dataset["out"] / "exp_001_temperature_csv"
    assert csv_dir.is_dir()
    assert len(list(csv_dir.glob("frame_*.csv"))) == 5
    assert (dataset["out"] / "campaign_b" / "exp_003_temperature_csv").is_dir()


# -- resume / skip logic -----------------------------------------------------------
def test_second_run_skips_completed_recordings(dataset):
    first = run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                                   formats=("h5",)))
    out = dataset["out"] / "exp_001_temperature.h5"
    stamp = out.stat().st_mtime_ns

    second = run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                                    formats=("h5",)))
    assert second.converted == 0
    assert second.skipped == 3
    assert out.stat().st_mtime_ns == stamp, "a skipped file must not be rewritten"
    assert first.frames == second.frames_skipped


def test_changed_recording_is_converted_again(dataset):
    run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                           formats=("h5",)))
    target = dataset["good"][0]
    extra = _make(dataset["input"], "exp_001.csq", frames=6)
    assert extra == target
    result = run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                                    formats=("h5",)))
    outcome = next(o for o in result.outcomes if o.input == target)
    assert outcome.status == "ok"
    assert outcome.frames == 6


def test_overwrite_forces_reconversion(dataset):
    run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                           formats=("h5",)))
    result = run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                                    formats=("h5",), overwrite=True))
    assert result.converted == 3
    assert result.skipped == 0


def test_settings_change_invalidates_the_skip(dataset):
    run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                           formats=("h5",)))
    result = run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                                    formats=("h5",), mode="apparent"))
    assert result.converted == 3, "a different conversion mode must not be skipped"


def test_deleted_output_is_rebuilt(dataset):
    run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                           formats=("h5",)))
    (dataset["out"] / "exp_002_temperature.h5").unlink()
    result = run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                                    formats=("h5",)))
    assert (dataset["out"] / "exp_002_temperature.h5").is_file()
    outcome = next(o for o in result.outcomes if o.input.name == "exp_002.csq")
    assert outcome.status == "ok"


def test_unreadable_state_file_does_not_abort_the_run(dataset):
    dataset["out"].mkdir(parents=True, exist_ok=True)
    (dataset["out"] / ".flir_csq_state.json").write_text("{not json")
    result = run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                                    formats=("h5",)))
    assert result.converted == 3
    state = load_state(dataset["out"])
    assert len(state["files"]) == 4


# -- dry run, naming, layout -------------------------------------------------------
def test_dry_run_writes_no_cube(dataset):
    result = run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                                    formats=("h5",), dry_run=True))
    assert result.planned == 4
    assert result.converted == 0
    assert not list(dataset["out"].glob("*.h5"))


def test_stem_naming_and_flat_output(dataset):
    result = run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                                    formats=("h5",), naming="stem", preserve_tree=False))
    assert result.converted == 3
    assert (dataset["out"] / "exp_001.h5").is_file()
    assert (dataset["out"] / "exp_003.h5").is_file()


def test_duplicate_file_names_in_different_directories_do_not_collide(tmp_path):
    inp = tmp_path / "in"
    _make(inp / "day1", "run.csq", frames=3)
    _make(inp / "day2", "run.csq", frames=5)
    result = run_batch(BatchOptions(input_dir=inp, output_dir=tmp_path / "out",
                                    formats=("h5",), preserve_tree=False))
    assert result.converted == 2
    files = sorted((tmp_path / "out").rglob("run_temperature.h5"))
    assert len(files) == 2
    shapes = []
    for path in files:
        with h5py.File(path) as f:
            shapes.append(f["temperature_C"].shape[0])
    assert sorted(shapes) == [3, 5]


# -- artefacts ---------------------------------------------------------------------
def test_state_log_and_summary_are_written(dataset):
    result = run_batch(BatchOptions(input_dir=dataset["input"], output_dir=dataset["out"],
                                    formats=("h5",)))
    state = json.loads((dataset["out"] / ".flir_csq_state.json").read_text())
    assert state["files"]["exp_001.csq"]["status"] == "ok"
    assert state["files"]["exp_broken.csq"]["status"] == "failed"
    assert result.log_file.is_file()
    text = result.log_file.read_text()
    assert "exp_001.csq" in text and "exp_broken.csq" in text
    summary = json.loads((dataset["out"] / "batch_summary.json").read_text())
    assert summary["totals"]["converted"] == 3
    assert summary["totals"]["failed"] == 1
    assert summary["totals"]["frames_written"] == 10
    assert {f["status"] for f in summary["files"]} == {"ok", "failed"}


def test_batch_does_not_import_qt():
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); import flir_csq.batch; "
            "print('PySide6' in sys.modules)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=True)
    assert proc.stdout.strip() == "False", "headless batch must not pull in Qt"


# -- CLI ---------------------------------------------------------------------------
def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True,
                          timeout=300, check=False)


def test_cli_batch_interface(dataset):
    proc = _cli("--input-dir", str(dataset["input"]), "--output-dir", str(dataset["out"]),
                "--format", "h5")
    assert proc.returncode == 4, proc.stderr  # one corrupt file, three converted
    assert (dataset["out"] / "exp_001_temperature.h5").is_file()
    assert "batch finished: 3 converted, 0 skipped, 1 failed" in proc.stdout
    assert "FAILED" in proc.stdout
    assert "log " in proc.stdout


def test_cli_batch_json_summary(dataset):
    proc = _cli("--input-dir", str(dataset["input"]), "--output-dir", str(dataset["out"]),
                "--format", "h5", "--json")
    assert proc.returncode == 4
    payload = json.loads(proc.stdout)
    assert payload["totals"]["converted"] == 3
    assert payload["exit_code"] == 4
    assert {f["status"] for f in payload["files"]} == {"ok", "failed"}


def test_cli_batch_dry_run(dataset):
    proc = _cli("--input-dir", str(dataset["input"]), "--output-dir", str(dataset["out"]),
                "--dry-run")
    assert proc.returncode == 0
    assert "would do" in proc.stdout
    assert not list(dataset["out"].glob("*.h5"))


def test_cli_rejects_input_and_input_dir_together(dataset):
    proc = _cli(str(dataset["good"][0]), "--input-dir", str(dataset["input"]))
    assert proc.returncode != 0
    assert "not both" in (proc.stderr + proc.stdout)


def test_cli_returns_2_when_no_recordings_found(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = _cli("--input-dir", str(empty), "--output-dir", str(tmp_path / "out"))
    assert proc.returncode == 2
    assert "no recordings found" in proc.stdout
