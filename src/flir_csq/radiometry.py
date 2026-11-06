"""Radiometry: sensor counts -> calibrated temperature in degrees Celsius.

The conversion is the standard FLIR radiometric model, applied with the
constants stored in the recording itself. No universal scale factor, no
coefficients borrowed from another camera.

Planck inversion (``R1``, ``B``, ``F``, ``O``, ``R2`` from each frame's
CameraInfo record, ExifTool's ``Planck*`` tags)::

    T[K] = B / ln( R1 / (R2 * (raw + O)) + F )

The camera measures band-integrated radiance, not a blackbody temperature, so
when the recording also carries the measurement parameters the counts are
corrected for emissivity, reflected apparent temperature, atmospheric
transmission and an IR window before inversion. In linear (raw-equivalent)
signal units ``c = raw + O``:

    c_meas = tau_win * [ eps*tau*c_obj + (1-eps)*tau*c_refl + (1-tau)*c_atm ]
             + (1-tau_win)*c_win

    c_obj  = [ c_meas/tau_win - (1-tau)*c_atm - (1-tau_win)/tau_win*c_win ]
             / (eps*tau) - (1-eps)/eps*c_refl

with ``c_T = R1 / (R2*(exp(B/(T+273.15)) - F))``. This is the formulation of
Minkina & Dudzik as implemented by ThermImage/flirpy; it reduces exactly to the
Planck inversion when ``eps = tau = tau_win = 1``.

Atmospheric transmittance uses the camera's own ``AtmosphericTrans*``
coefficients with the water-vapour fit of the same literature::

    w   = RH * exp(1.5587 + 0.06939*Ta - 2.7816e-4*Ta^2 + 6.8455e-7*Ta^3)
    tau = X*exp(-sqrt(d/2)*(a1 + b1*sqrt(w)))
        + (1-X)*exp(-sqrt(d/2)*(a2 + b2*sqrt(w)))

It assumes a homogeneous layer with the attenuating path split in two halves and
a grey (band-averaged) atmosphere. It is the model FLIR's own tools use, not an
exact line-by-line radiative transfer solution; see ``docs/calibration.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import MissingCalibration
from .metadata import CameraInfo

#: Documented FLIR defaults for the atmospheric model (ExifTool FLIR.pm comments).
DEFAULT_ATMOSPHERIC = {
    "alpha1": 0.006569,
    "alpha2": 0.012620,
    "beta1": -0.002276,
    "beta2": -0.006670,
    "x": 1.900000,
}

MODES = ("auto", "apparent", "object")


@dataclass(frozen=True)
class PlanckCoefficients:
    """The five constants needed to invert the camera's calibration."""

    r1: float
    b: float
    f: float
    o: float
    r2: float
    source: str = "frame-metadata"

    def __post_init__(self) -> None:
        if not self.r1 or not self.r2 or not self.b:
            raise MissingCalibration("Planck constants R1, R2 and B must be non-zero")

    @classmethod
    def from_camera_info(cls, info: CameraInfo) -> PlanckCoefficients:
        missing = [name for name in ("planck_r1", "planck_b", "planck_f", "planck_o", "planck_r2")
                   if getattr(info, name) is None]
        if missing:
            raise MissingCalibration(
                "the recording does not carry a usable radiometric calibration",
                hint=(
                    "Missing Planck constant(s): " + ", ".join(missing) + ". Without them no "
                    "physically meaningful temperature can be produced; export raw counts only "
                    "(--mode raw / --save-raw) or re-record radiometrically."
                ),
            )
        return cls(r1=float(info.planck_r1), b=float(info.planck_b), f=float(info.planck_f),
                   o=float(info.planck_o), r2=float(info.planck_r2))

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> PlanckCoefficients:
        """Build from a ``planck_r1``/``planck_b``/... mapping (metadata / test data)."""
        return cls(r1=float(values["planck_r1"]), b=float(values["planck_b"]),
                   f=float(values["planck_f"]), o=float(values["planck_o"]),
                   r2=float(values["planck_r2"]))

    def as_dict(self) -> dict[str, float]:
        return {"planck_r1": self.r1, "planck_b": self.b, "planck_f": self.f,
                "planck_o": self.o, "planck_r2": self.r2}


@dataclass
class MeasurementParameters:
    """Everything the object-temperature correction needs, as found in the file."""

    emissivity: float | None = None
    object_distance_m: float | None = None
    reflected_apparent_temperature_c: float | None = None
    atmospheric_temperature_c: float | None = None
    relative_humidity: float | None = None
    ir_window_transmission: float | None = None
    ir_window_temperature_c: float | None = None
    atmospheric_transmission: float | None = None
    source: str = "frame-metadata"

    def missing_for_object_correction(self) -> list[str]:
        required = {
            "emissivity": self.emissivity,
            "reflected_apparent_temperature_c": self.reflected_apparent_temperature_c,
            "atmospheric_temperature_c": self.atmospheric_temperature_c,
        }
        return [k for k, v in required.items() if v is None]

    def as_dict(self) -> dict[str, Any]:
        return {
            "emissivity": self.emissivity,
            "object_distance_m": self.object_distance_m,
            "reflected_apparent_temperature_c": self.reflected_apparent_temperature_c,
            "atmospheric_temperature_c": self.atmospheric_temperature_c,
            "relative_humidity": self.relative_humidity,
            "ir_window_transmission": self.ir_window_transmission,
            "ir_window_temperature_c": self.ir_window_temperature_c,
            "atmospheric_transmission": self.atmospheric_transmission,
            "source": self.source,
        }


@dataclass
class ConversionInfo:
    """What the conversion actually did - written into the output metadata."""

    mode_requested: str
    mode_effective: str
    coefficients: PlanckCoefficients
    parameters: MeasurementParameters
    atmospheric_transmission: float | None = None
    warnings: list[str] = field(default_factory=list)
    invalid_pixels: int = 0
    total_pixels: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode_requested": self.mode_requested,
            "mode_effective": self.mode_effective,
            "corrections_applied": self.mode_effective == "object",
            "coefficients": self.coefficients.as_dict(),
            "parameters": self.parameters.as_dict(),
            "atmospheric_transmission": self.atmospheric_transmission,
            "invalid_pixels_last_frame": self.invalid_pixels,
            "warnings": list(self.warnings),
        }


def measurement_parameters_from(info: CameraInfo) -> MeasurementParameters:
    """Copy the measurement parameters a frame carries into a plain object."""
    params = MeasurementParameters(
        emissivity=info.emissivity,
        object_distance_m=info.object_distance_m,
        reflected_apparent_temperature_c=info.reflected_apparent_temperature_c,
        atmospheric_temperature_c=info.atmospheric_temperature_c,
        relative_humidity=info.relative_humidity,
        ir_window_transmission=info.ir_window_transmission,
        ir_window_temperature_c=info.ir_window_temperature_c,
    )
    if None not in (info.object_distance_m, info.atmospheric_temperature_c, info.relative_humidity):
        params.atmospheric_transmission = atmospheric_transmission(
            float(info.object_distance_m),
            float(info.atmospheric_temperature_c),
            float(info.relative_humidity),
            _atmospheric_coefficients(info),
        )
    return params


def _atmospheric_coefficients(info: CameraInfo) -> dict[str, float]:
    coeffs = dict(DEFAULT_ATMOSPHERIC)
    for key, attr in (("alpha1", "atmospheric_trans_alpha1"), ("alpha2", "atmospheric_trans_alpha2"),
                      ("beta1", "atmospheric_trans_beta1"), ("beta2", "atmospheric_trans_beta2"),
                      ("x", "atmospheric_trans_x")):
        value = getattr(info, attr)
        if value is not None:
            coeffs[key] = float(value)
    return coeffs


def atmospheric_transmission(distance_m: float, air_temperature_c: float,
                             relative_humidity: float,
                             coefficients: dict[str, float] | None = None) -> float:
    """Band-averaged transmittance of the air path (0..1), never exactly 0."""
    c = dict(DEFAULT_ATMOSPHERIC)
    if coefficients:
        for key, value in coefficients.items():
            if key in c and value is not None:
                c[key] = float(value)
    rh = float(relative_humidity)
    if rh > 1.5:  # 0..100 given as percent
        rh /= 100.0
    rh = min(max(rh, 0.0), 1.0)
    ta = float(air_temperature_c)
    water = rh * math.exp(1.5587 + 0.06939 * ta - 2.7816e-4 * ta ** 2 + 6.8455e-7 * ta ** 3)
    root = math.sqrt(max(float(distance_m), 0.0) / 2.0)
    sqrt_w = math.sqrt(max(water, 0.0))
    tau = c["x"] * math.exp(-root * (c["alpha1"] + c["beta1"] * sqrt_w)) + (
        1.0 - c["x"]
    ) * math.exp(-root * (c["alpha2"] + c["beta2"] * sqrt_w))
    return float(min(max(tau, 1e-4), 1.0))


def signal_from_temperature_c(temperature_c: float | np.ndarray,
                              coef: PlanckCoefficients) -> Any:
    """Raw-equivalent radiance ``c_T`` of a blackbody at ``temperature_c``."""
    t_k = np.asarray(temperature_c, dtype=np.float64) + 273.15
    return coef.r1 / (coef.r2 * (np.exp(coef.b / t_k) - coef.f))


def temperature_from_signal_c(signal: np.ndarray, coef: PlanckCoefficients) -> np.ndarray:
    """Invert the Planck relation for raw-equivalent radiance ``c = raw + O``."""
    signal = np.asarray(signal, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inner = coef.r1 / (coef.r2 * signal) + coef.f
        t = coef.b / np.log(inner) - 273.15
    t[~np.isfinite(t)] = np.nan
    return t


def apparent_temperature_c(raw: np.ndarray, coef: PlanckCoefficients) -> np.ndarray:
    """Planck inversion of the counts as stored: uncorrected for emissivity/atmosphere."""
    return temperature_from_signal_c(np.asarray(raw, dtype=np.float64) + coef.o, coef)


def object_temperature_c(raw: np.ndarray, coef: PlanckCoefficients,
                         params: MeasurementParameters) -> np.ndarray:
    """Emissivity / reflected-temperature / atmosphere / window corrected temperature."""
    missing = params.missing_for_object_correction()
    if missing:
        raise MissingCalibration(
            "object-temperature correction is not possible: the recording lacks "
            + ", ".join(missing),
            hint="Use --mode apparent (Planck inversion only) or supply the parameters; the "
                 "difference can be several kelvin for low-emissivity surfaces.",
        )
    eps = float(params.emissivity)
    tau_win = float(params.ir_window_transmission if params.ir_window_transmission is not None else 1.0)
    tau = params.atmospheric_transmission
    if tau is None:
        if params.object_distance_m is not None and params.atmospheric_temperature_c is not None \
                and params.relative_humidity is not None:
            tau = atmospheric_transmission(float(params.object_distance_m),
                                           float(params.atmospheric_temperature_c),
                                           float(params.relative_humidity))
        else:
            tau = 1.0
    tau = float(min(max(tau, 1e-4), 1.0))

    c_meas = np.asarray(raw, dtype=np.float64) + coef.o
    c_refl = float(signal_from_temperature_c(float(params.reflected_apparent_temperature_c), coef))
    c_atm = float(signal_from_temperature_c(float(params.atmospheric_temperature_c), coef))
    numerator = c_meas / tau_win - (1.0 - tau) * c_atm
    if tau_win < 1.0 and params.ir_window_temperature_c is not None:
        c_win = float(signal_from_temperature_c(float(params.ir_window_temperature_c), coef))
        numerator = numerator - (1.0 - tau_win) / tau_win * c_win
    c_obj = numerator / (eps * tau) - (1.0 - eps) / eps * c_refl
    with np.errstate(invalid="ignore", divide="ignore"):
        c_obj = np.where(c_obj > 0, c_obj, np.nan)
    return temperature_from_signal_c(c_obj, coef)


def invalid_mask(raw: np.ndarray, info: CameraInfo) -> np.ndarray:
    """True where a pixel carries no usable measurement.

    Pixels outside the calibration range are physically invalid: inverting them
    extrapolates the calibration and produces numbers that look like data but
    are not. They become NaN in the output rather than being clipped.
    """
    raw = np.asarray(raw)
    mask = (raw == 0) | (raw == 65535)
    lo, hi = info.raw_value_range_min, info.raw_value_range_max
    if lo:
        mask |= raw < lo
    if hi:
        mask |= raw > hi
    return mask


def convert_frame(raw: np.ndarray, info: CameraInfo, *, mode: str = "auto",
                  params: MeasurementParameters | None = None,
                  coef: PlanckCoefficients | None = None) -> tuple[np.ndarray, ConversionInfo]:
    """Convert one frame's counts to degrees Celsius.

    ``mode``: ``apparent`` = Planck inversion only, ``object`` = fully corrected,
    ``auto`` = corrected when the recording carries every parameter the
    correction needs, otherwise apparent with an explicit warning.
    """
    if mode not in MODES:
        raise ValueError(f"unknown conversion mode {mode!r}; expected one of {MODES}")
    coefficients = coef or PlanckCoefficients.from_camera_info(info)
    parameters = params or measurement_parameters_from(info)
    warnings: list[str] = []
    effective = mode

    if mode in ("auto", "object"):
        missing = parameters.missing_for_object_correction()
        if missing:
            if mode == "object":
                raise MissingCalibration(
                    "object-temperature correction requested but the recording lacks "
                    + ", ".join(missing),
                    hint="Re-run with --mode apparent to use the Planck inversion only.",
                )
            effective = "apparent"
            warnings.append(
                "object correction not applied (missing " + ", ".join(missing) +
                "); temperature is the Planck apparent temperature"
            )
        else:
            effective = "object"
            if parameters.atmospheric_transmission is not None and \
                    parameters.atmospheric_transmission < 0.999:
                warnings.append(
                    f"atmospheric transmission {parameters.atmospheric_transmission:.4f} "
                    f"applied for {parameters.object_distance_m:.2f} m path"
                )

    mask = invalid_mask(raw, info)
    if effective == "object":
        temp = object_temperature_c(raw, coefficients, parameters)
    else:
        temp = apparent_temperature_c(raw, coefficients)
    temp = np.where(mask, np.nan, temp)
    out = temp.astype(np.float32)

    return out, ConversionInfo(
        mode_requested=mode,
        mode_effective=effective,
        coefficients=coefficients,
        parameters=parameters,
        atmospheric_transmission=parameters.atmospheric_transmission,
        warnings=warnings,
        invalid_pixels=int(mask.sum()),
        total_pixels=int(mask.size),
    )


def frame_statistics(temp: np.ndarray) -> dict[str, float | int | None]:
    """Summary numbers per frame, computed with NaN awareness."""
    arr = np.asarray(temp, dtype=np.float64)
    finite = np.isfinite(arr)
    n = int(arr.size)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return {"min_C": None, "max_C": None, "mean_C": None, "std_C": None,
                "valid_fraction": 0.0, "n_valid": 0, "n_total": n}
    vals = arr[finite]
    return {
        "min_C": float(vals.min()),
        "max_C": float(vals.max()),
        "mean_C": float(vals.mean()),
        "std_C": float(vals.std()),
        "valid_fraction": n_finite / n,
        "n_valid": n_finite,
        "n_total": n,
    }
