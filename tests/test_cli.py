"""Command-line interface tests (run as a real subprocess)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "convert_csq.py"


def run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False,
                          cwd=str(cwd or ROOT), timeout=300)


def test_help_lists_the_main_options():
    proc = run("--help")
    assert proc.returncode == 0
    for flag in ("--output", "--format", "--save-raw", "--save-metadata", "--inspect", "--demo",
                 "--verify", "--mode"):
        assert flag in proc.stdout


def test_no_arguments_prints_help():
    proc = run()
    assert proc.returncode == 2
    assert "usage:" in proc.stdout


def test_default_conversion_creates_hdf5_next_to_input(jpegls_recording):
    proc = run(str(jpegls_recording["path"]))
    assert proc.returncode == 0, proc.stderr
    expected = Path(jpegls_recording["path"]).with_name("jpegls_t540_temperature.h5")
    assert expected.is_file()
    assert "converted  8 frames" in proc.stdout
    assert "verify     PASS" in proc.stdout


def test_explicit_output_directory_and_formats(jpegls_recording, out_dir):
    proc = run(str(jpegls_recording["path"]), "--output", str(out_dir),
               "--format", "h5,npy,csv", "--save-raw", "--save-metadata")
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "jpegls_t540_temperature.h5").is_file()
    assert (out_dir / "jpegls_t540_temperature.npy").is_file()
    assert (out_dir / "jpegls_t540_temperature_csv").is_dir()
    assert (out_dir / "jpegls_t540_conversion.json").is_file()


def test_inspect_json_is_machine_readable(jpegls_recording):
    proc = run(str(jpegls_recording["path"]), "--inspect", "--json")
    assert proc.returncode == 0, proc.stderr
    info = json.loads(proc.stdout)
    assert info["frames"]["complete"] == 8
    assert info["dimensions"]["width"] == 64
    assert info["camera"]["camera_model"] == "FLIR T540"


def test_inspect_human_output(jpegls_recording):
    proc = run(str(jpegls_recording["path"]), "--inspect")
    assert proc.returncode == 0
    assert "FLIR T540" in proc.stdout
    assert "JPEG-LS" in proc.stdout
    assert "calibrated" in proc.stdout


def test_raw_only_exports_counts_when_calibration_is_missing(uncalibrated_recording, out_dir):
    proc = run(str(uncalibrated_recording["path"]), "--output", str(out_dir), "--raw-only")
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "nocal_temperature.h5").is_file()


def test_missing_calibration_exits_with_code_3(uncalibrated_recording, out_dir):
    proc = run(str(uncalibrated_recording["path"]), "--output", str(out_dir))
    assert proc.returncode == 3
    assert "Planck calibration" in proc.stderr
    assert "--raw-only" in proc.stderr


def test_verify_command_reports_pass(jpegls_recording, out_dir):
    run(str(jpegls_recording["path"]), "--output", str(out_dir))
    proc = run("--verify", str(out_dir / "jpegls_t540_temperature.h5"))
    assert proc.returncode == 0
    assert proc.stdout.startswith("PASS")


def test_verify_command_reports_missing_file(tmp_path):
    proc = run("--verify", str(tmp_path / "nope.h5"))
    assert proc.returncode == 3
    assert "FAIL" in proc.stdout


def test_demo_builds_a_synthetic_t540_recording_and_converts_it(tmp_path):
    demo = tmp_path / "demo.csq"
    proc = run("--demo", str(demo), "--output", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert demo.is_file()
    assert (tmp_path / "demo_temperature.h5").is_file()
    assert "464" in proc.stdout or "shape (12, 348, 464)" in proc.stdout


def test_bad_file_exits_with_code_2(tmp_path):
    bogus = tmp_path / "bogus.csq"
    bogus.write_bytes(b"not a recording" * 50)
    proc = run(str(bogus))
    assert proc.returncode == 2
    assert "error:" in proc.stderr


def test_unknown_format_is_rejected(jpegls_recording, tmp_path):
    proc = run(str(jpegls_recording["path"]), "--output", str(tmp_path), "--format", "parquet")
    assert proc.returncode != 0
    assert "unknown format" in (proc.stderr + proc.stdout)


def test_quiet_mode_prints_only_the_summary(jpegls_recording, tmp_path):
    proc = run(str(jpegls_recording["path"]), "--output", str(tmp_path), "--quiet")
    assert proc.returncode == 0
    assert "converted" in proc.stdout
