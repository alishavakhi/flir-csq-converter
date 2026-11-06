# Calibration and temperature conversion

Everything below is implemented in [`src/flir_csq/radiometry.py`](../src/flir_csq/radiometry.py)
and exercised by [`tests/test_radiometry.py`](../tests/test_radiometry.py). The rule that governs
the whole module: **the calibration comes from the recording, per frame.** No constant in this
repository describes any particular camera.

## 1. What the camera stores

The sensor produces a 14-bit measurement count per pixel (`raw`). That count is linear in the
band-integrated radiance the pixel received, but it is *not* a temperature: converting it requires
the camera's Planck fit and the measurement parameters that were active at capture time. Both are
stored in the frame's CameraInfo record:

| field | record offset | used as |
|---|---|---|
| `planck_r1` | 0x58 | Planck R1 |
| `planck_b` | 0x5C | Planck B [K] |
| `planck_f` | 0x60 | Planck F |
| `planck_o` | 0x308 | Planck O (int32) |
| `planck_r2` | 0x30C | Planck R2 |
| `emissivity` | 0x20 | ε |
| `object_distance_m` | 0x24 | d [m] |
| `reflected_apparent_temperature` | 0x28 | T_refl [K] |
| `atmospheric_temperature` | 0x2C | T_atm [K] |
| `ir_window_temperature` | 0x30 | T_win [K] |
| `ir_window_transmission` | 0x34 | τ_win |
| `relative_humidity` | 0x3C | RH (0–1, or 0–100 if the camera writes percent) |
| `atmospheric_trans_alpha1/2, beta1/2, x` | 0x70–0x80 | τ model |
| `raw_value_range_min/max` | 0x310/0x312 | validity limits |

Offsets are ExifTool's `%Image::ExifTool::FLIR::CameraInfo` table (see `format.md`).

## 2. Planck inversion (apparent temperature)

```
T[K] = B / ln( R1 / (R2 · (raw + O)) + F )
T[°C] = T[K] − 273.15
```

This is exact for the camera's own fit and is what `--mode apparent` writes. It assumes the pixel
saw a blackbody of emissivity 1 at zero distance in a perfectly transparent atmosphere — so it is
the right number to use only when those assumptions hold, and it is never a substitute for the
corrected value when the file carries the parameters.

## 3. Object temperature (default, `--mode auto` / `object`)

Radiance is additive in the linear sensor domain. With `c = raw + O` (raw-equivalent radiance) and
`c_T = R1 / (R2 · (exp(B/(T+273.15)) − F))` for a blackbody at `T`, the camera receives

```
c_meas = τ_win · [ ε·τ·c_obj + (1−ε)·τ·c_refl + (1−τ)·c_atm ] + (1−τ_win)·c_win
```

which inverts to

```
c_obj = [ c_meas/τ_win − (1−τ)·c_atm − (1−τ_win)/τ_win·c_win ] / (ε·τ)
        − (1−ε)/ε · c_refl
```

and then `T_obj = B / ln(R1/(R2·c_obj) + F) − 273.15`. With `ε = τ = τ_win = 1` this collapses
exactly to the Planck inversion (a test asserts equality to 1e-9 °C). This is Minkina & Dudzik's
formulation as implemented by ThermImage and flirpy; `tests/test_radiometry.py` cross-checks the
`ε < 1, τ = 1` case against `flirpy.util.raw.raw2temp` to 1e-6 °C.

### Atmospheric transmission

FLIR's band-averaged empirical fit, with the coefficients stored in the recording (documented
defaults are used only when a coefficient is absent):

```
w   = RH · exp(1.5587 + 0.06939·T_a − 2.7816e-4·T_a² + 6.8455e-7·T_a³)      [mmHg]
τ   = X · exp(−√(d/2)·(α₁ + β₁·√w)) + (1−X) · exp(−√(d/2)·(α₂ + β₂·√w))
```

`d` is the object distance, `T_a` the atmospheric temperature in °C, `RH` the relative humidity as
a fraction. `τ` is clamped to `[1e-4, 1]`.

**Modelling choice, stated openly:** the expression above is FLIR's full-path transmittance and is
applied once. flirpy/ThermImage treat the same expression as a *half*-path transmittance and apply
it twice (plus a second atmosphere term), which over-corrects slightly. On the public T1020 sample
(20 m, ε = 1) the two differ by at most **0.21 °C**, mean 0.04 °C. Which one matches FLIR Research
Studio can only be settled with a Research Studio comparison on your own recording — see
`validation.md`.

## 4. When the correction is applied

| mode | behaviour |
|---|---|
| `auto` (default) | corrects when ε, T_refl and T_atm are present (τ from d/RH when available, otherwise τ = 1 with a note); otherwise writes the Planck apparent temperature and records a warning in the metadata |
| `apparent` | Planck inversion only; no correction, whatever the file says |
| `object` | correction demanded; if a required parameter is missing the conversion **fails** with the parameter named |

The mode actually used is written to `/temperature_C`'s parent metadata as
`conversion.mode_effective`, together with every coefficient and parameter, so a temperature can
always be traced back to the numbers that produced it.

## 5. Invalid pixels: NaN, never a clipped number

A pixel becomes `NaN` when

* `raw == 0` or `raw == 65535` (rail values, no measurement),
* `raw` is outside `raw_value_range_min … raw_value_range_max` (outside the camera's stated
  calibration range — inverting it extrapolates the fit),
* the inversion is not finite (non-positive radiance, log domain error).

NaN is deliberate: a saturated defect is *missing data*, not "the maximum of the range". Downstream
code must use `np.nanmin`, `np.nanmean`, etc., or `--save-mask` to get an explicit `valid_mask`.
`valid_mask` is also derivable from `np.isfinite(temperature_C)`.

## 6. Units and precision

* Output is **always** °C (`temperature_C`, attribute `unit="degC"`). Kelvin appears only inside
  the conversion, and the JSON metadata states the method explicitly.
* Storage is float32: at 20 °C that is ~2e-6 °C resolution, far below the camera's noise
  equivalent temperature difference (~0.03 K for a T540-class microbolometer). float64 would add
  file size for no information.
* Raw counts are kept as uint16 exactly as decoded, so nothing is lost even if a future decoder or
  calibration revision changes the temperature.

## 7. Uncertainty — what these numbers are and are not

The output is the temperature implied by **the calibration and measurement parameters stored in
the file**, under a band-averaged atmospheric model. It is not:

* a correction for a wrong emissivity setting on the camera (the report shows ε; override it with
  `--emissivity` and the metadata records the override);
* a model of a non-diffuse or non-uniform background (the reflected temperature is a single value);
* a spectral correction — the camera's spectral response is treated as flat within its band;
* a substitute for a blackbody calibration check. If absolute accuracy matters, compare against a
  reference blackbody or FLIR Research Studio on the same recording.

Reported accuracy of a defect measurement therefore also depends on the spatial calibration
(distance, lens, FOV), which this tool records but does not use to convert pixels to millimetres.
