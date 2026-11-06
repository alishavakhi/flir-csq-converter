"""Shared fixtures: synthetic recordings with known ground truth."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flir_csq.synthetic import SyntheticCamera, make_field, make_synthetic_csq  # noqa: E402


@pytest.fixture(scope="session")
def jpegls_recording(tmp_path_factory) -> dict:
    """8 frames, JPEG-LS payload, 30 Hz camera clock, emissivity 1.0."""
    path = tmp_path_factory.mktemp("csq") / "jpegls_t540.csq"
    temps = make_field(8, 48, 64, base_c=26.0, amplitude_c=9.0)
    return make_synthetic_csq(path, n_frames=8, height=48, width=64,
                              compression="jpegls", temperatures=temps)


@pytest.fixture(scope="session")
def emissive_recording(tmp_path_factory) -> dict:
    """Object-temperature test case: eps = 0.92, 1.0 m, RH 60 %."""
    path = tmp_path_factory.mktemp("csq") / "emissive.csq"
    camera = SyntheticCamera(emissivity=0.92, object_distance_m=1.0,
                             reflected_temperature_c=22.0, atmospheric_temperature_c=24.0,
                             relative_humidity=0.6, width=40, height=30)
    temps = make_field(5, 30, 40, base_c=30.0, amplitude_c=6.0)
    return make_synthetic_csq(path, n_frames=5, height=30, width=40,
                              compression="raw", camera=camera, temperatures=temps)


@pytest.fixture(scope="session")
def uncalibrated_recording(tmp_path_factory) -> dict:
    """Same container, but without Planck constants."""
    path = tmp_path_factory.mktemp("csq") / "nocal.csq"
    camera = SyntheticCamera(with_planck=False, width=16, height=12)
    temps = make_field(3, 12, 16)
    return make_synthetic_csq(path, n_frames=3, height=12, width=16,
                              compression="raw", camera=camera, temperatures=temps)


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    d = tmp_path / "out"
    d.mkdir()
    return d


@pytest.fixture
def small_recording(tmp_path: Path) -> dict:
    """Fresh small recording per test (safe to truncate or corrupt in place)."""
    path = tmp_path / "small.csq"
    temps = make_field(4, 12, 16, base_c=24.0, amplitude_c=5.0)
    return make_synthetic_csq(path, n_frames=4, height=12, width=16,
                              compression="jpegls", temperatures=temps)
