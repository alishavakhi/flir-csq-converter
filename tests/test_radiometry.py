"""Radiometry tests: Planck inversion, corrections, units, failure modes."""

from __future__ import annotations

import numpy as np
import pytest

from flir_csq.errors import MissingCalibration
from flir_csq.metadata import CameraInfo
from flir_csq.radiometry import (
    DEFAULT_ATMOSPHERIC,
    MeasurementParameters,
    PlanckCoefficients,
    apparent_temperature_c,
    atmospheric_transmission,
    convert_frame,
    frame_statistics,
    invalid_mask,
    object_temperature_c,
    signal_from_temperature_c,
    temperature_from_signal_c,
)
from flir_csq.synthetic import DEMO_CALIBRATION, SyntheticCamera, temperatures_to_counts

COEF = PlanckCoefficients.from_dict(DEMO_CALIBRATION)


def test_planck_inversion_of_forward_model_is_exact():
    temps = np.array([[-20.0, 0.0, 21.5], [37.0, 100.0, 150.0]])
    signal = signal_from_temperature_c(temps, COEF)
    back = temperature_from_signal_c(signal, COEF)
    np.testing.assert_allclose(back, temps, atol=1e-9)


def test_apparent_temperature_is_in_celsius():
    """0 degC must come out as 0, not 273.15 and not a count."""
    signal = signal_from_temperature_c(0.0, COEF)
    raw = signal - COEF.o
    assert apparent_temperature_c(np.array([[raw]]), COEF)[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_planck_inversion_is_not_linear():
    """A doubling of counts must not double the temperature (guards against scale factors)."""
    raw_lo, raw_hi = 13000.0, 26000.0
    t_lo = float(apparent_temperature_c(np.array([[raw_lo]]), COEF)[0, 0])
    t_hi = float(apparent_temperature_c(np.array([[raw_hi]]), COEF)[0, 0])
    assert t_hi > t_lo
    assert abs(t_hi - 2 * t_lo) > 10


def test_object_correction_recovers_the_forward_model():
    """eps < 1: the corrected temperature must return the temperatures used to build the counts."""
    camera = SyntheticCamera(emissivity=0.9, reflected_temperature_c=25.0,
                             atmospheric_temperature_c=22.0, object_distance_m=1.0,
                             relative_humidity=0.5)
    truth = np.array([[10.0, 30.0], [45.0, 80.0]])
    counts = temperatures_to_counts(truth, camera)
    info = CameraInfo(emissivity=0.9, reflected_apparent_temperature_c=25.0,
                      atmospheric_temperature_c=22.0, object_distance_m=1.0,
                      relative_humidity=0.5, **DEMO_CALIBRATION)
    params = MeasurementParameters(emissivity=0.9, reflected_apparent_temperature_c=25.0,
                                   atmospheric_temperature_c=22.0, object_distance_m=1.0,
                                   relative_humidity=0.5, ir_window_transmission=1.0)
    out, meta = convert_frame(counts, info, mode="object", params=params)
    assert meta.mode_effective == "object"
    # uint16 quantisation of the counts is the only error source (~1 count ~ 0.01 degC)
    np.testing.assert_allclose(out, truth, atol=0.02)
    apparent = apparent_temperature_c(counts, COEF)
    assert np.max(np.abs(apparent - truth)) > 1.0, "emissivity correction must matter"


def test_object_correction_equals_planck_inversion_for_a_blackbody():
    raw = np.array([[13000, 14000, 15000]], dtype=np.uint16)
    params = MeasurementParameters(emissivity=1.0, reflected_apparent_temperature_c=20.0,
                                   atmospheric_temperature_c=20.0, object_distance_m=0.0,
                                   relative_humidity=0.0, ir_window_transmission=1.0,
                                   atmospheric_transmission=1.0)
    object_c = object_temperature_c(raw, COEF, params)
    apparent = apparent_temperature_c(raw, COEF)
    np.testing.assert_allclose(object_c, apparent, atol=1e-9)


def test_atmospheric_transmission_decreases_with_distance_and_stays_bounded():
    near = atmospheric_transmission(0.0, 20.0, 0.5, DEFAULT_ATMOSPHERIC)
    mid = atmospheric_transmission(10.0, 20.0, 0.5, DEFAULT_ATMOSPHERIC)
    far = atmospheric_transmission(100.0, 20.0, 0.5, DEFAULT_ATMOSPHERIC)
    assert near == pytest.approx(1.0, abs=1e-9)
    assert 0 < far < mid < near <= 1.0
    assert atmospheric_transmission(20.0, 20.0, 0.4) < 0.999


def test_humidity_may_be_fraction_or_percent():
    a = atmospheric_transmission(50.0, 25.0, 0.6)
    b = atmospheric_transmission(50.0, 25.0, 60.0)
    assert a == pytest.approx(b, rel=1e-9)


def test_missing_parameters_raise_instead_of_being_guessed():
    info = CameraInfo(emissivity=None, reflected_apparent_temperature_c=None,
                      atmospheric_temperature_c=None, **DEMO_CALIBRATION)
    with pytest.raises(MissingCalibration):
        convert_frame(np.full((2, 2), 14000, np.uint16), info, mode="object")


def test_auto_mode_falls_back_to_apparent_with_a_warning():
    info = CameraInfo(emissivity=None, reflected_apparent_temperature_c=None,
                      atmospheric_temperature_c=None, **DEMO_CALIBRATION)
    _, meta = convert_frame(np.full((2, 2), 14000, np.uint16), info, mode="auto")
    assert meta.mode_effective == "apparent"
    assert any("object correction not applied" in w for w in meta.warnings)


def test_missing_planck_constants_are_reported_by_name():
    info = CameraInfo(planck_r1=None, planck_b=1328.9, planck_f=1.0, planck_o=-3869.0,
                      planck_r2=0.0135)
    with pytest.raises(MissingCalibration) as exc:
        convert_frame(np.zeros((2, 2), np.uint16), info)
    assert "planck_r1" in str(exc.value)


def test_calibration_comes_from_the_file_not_from_a_constant():
    """Two recordings of the same scene with different constants must both come out right."""
    truth = np.array([[20.0, 40.0]])
    # distance 0 / RH 0 -> tau = 1, so this test isolates the Planck constants
    camera_a = SyntheticCamera(calibration=dict(DEMO_CALIBRATION), object_distance_m=0.0,
                               relative_humidity=0.0)
    camera_b = SyntheticCamera(calibration={"planck_r1": 21106.77, "planck_b": 1501.0,
                                            "planck_f": 1.0, "planck_o": -7340,
                                            "planck_r2": 0.012545258},
                               object_distance_m=0.0, relative_humidity=0.0)
    counts_a = temperatures_to_counts(truth, camera_a)
    counts_b = temperatures_to_counts(truth, camera_b)
    assert abs(int(counts_a[0, 0]) - int(counts_b[0, 0])) > 100, "counts differ between cameras"
    for counts, camera in ((counts_a, camera_a), (counts_b, camera_b)):
        info = CameraInfo(emissivity=1.0, reflected_apparent_temperature_c=20.0,
                          atmospheric_temperature_c=20.0, object_distance_m=0.0,
                          relative_humidity=0.0, ir_window_transmission=1.0,
                          **camera.calibration)
        out, _ = convert_frame(counts, info, mode="apparent")
        np.testing.assert_allclose(out, truth, atol=0.02)


def test_invalid_pixels_become_nan():
    info = CameraInfo(raw_value_range_min=5589, raw_value_range_max=53132, **DEMO_CALIBRATION)
    raw = np.array([[0, 65535, 5588, 14000]], dtype=np.uint16)
    mask = invalid_mask(raw, info)
    assert mask.tolist() == [[True, True, True, False]]
    out, meta = convert_frame(raw, info, mode="apparent")
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 1]) and np.isnan(out[0, 2])
    assert np.isfinite(out[0, 3])
    assert meta.invalid_pixels == 3


def test_frame_statistics_ignore_nan():
    temp = np.array([[np.nan, 10.0], [20.0, 30.0]], dtype=np.float32)
    stats = frame_statistics(temp)
    assert stats["min_C"] == 10.0 and stats["max_C"] == 30.0
    assert stats["mean_C"] == pytest.approx(20.0)
    assert stats["valid_fraction"] == 0.75


def test_cross_check_against_flirpy():
    """Independent implementation (flirpy/ThermImage) must agree for a blackbody."""
    pytest.importorskip("flirpy")
    aw = pytest.importorskip("flirpy.util.raw")
    raw = np.array([[12800, 13300, 14100, 15500]], dtype=np.uint16)
    meta = {
        "Atmospheric Trans Alpha 1": "0.006569",
        "Atmospheric Trans Alpha 2": "0.012620",
        "Atmospheric Trans Beta 1": "-0.002276",
        "Atmospheric Trans Beta 2": "-0.006670",
        "Atmospheric Trans X": "1.900000",
        "Planck R1": str(DEMO_CALIBRATION["planck_r1"]),
        "Planck R2": str(DEMO_CALIBRATION["planck_r2"]),
        "Planck O": str(DEMO_CALIBRATION["planck_o"]),
        "Planck B": str(DEMO_CALIBRATION["planck_b"]),
        "Planck F": str(DEMO_CALIBRATION["planck_f"]),
        "Emissivity": "0.95",
        "IR Window Transmission": "1",
        "IR Window Temperature": "20.0",
        "Object Distance": "0.0",
        "Atmospheric Temperature": "20.0",
        "Reflected Apparent Temperature": "20.0",
        "Relative Humidity": "0.5",
    }
    reference = aw.raw2temp(raw, meta)
    params = MeasurementParameters(emissivity=0.95, reflected_apparent_temperature_c=20.0,
                                   atmospheric_temperature_c=20.0, object_distance_m=0.0,
                                   relative_humidity=0.5, ir_window_transmission=1.0,
                                   atmospheric_transmission=1.0)
    ours = object_temperature_c(raw, COEF, params)
    np.testing.assert_allclose(ours, reference, atol=1e-6)
