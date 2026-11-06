# FLIR CSQ → Temperature

Turn a FLIR T-series **`.csq` radiometric recording** into a numerical temperature cube:
`(frames, height, width)` in **°C**, written to HDF5, with per-frame camera timestamps and the
full calibration record. No JPEG extraction, no palette, no manual per-frame work: select the
file, press convert.

```
experiment_001.csq  ──►  experiment_001_temperature.h5
                         ├── /temperature_C        float32 (frames, H, W)   [°C]
                         ├── /raw_counts           uint16  (frames, H, W)   [sensor counts]
                         ├── /timestamps           float64 (frames,)        [s, camera clock]
                         ├── /timestamps_unix_ms   int64   (frames,)        [epoch ms, UTC]
                         └── /metadata             JSON: camera, calibration, conversion, warnings
```

```python
import h5py, numpy as np
with h5py.File("experiment_001_temperature.h5") as f:
    T = f["temperature_C"]          # (frames, 348, 464) float32, degrees Celsius
    print(T.shape, np.nanmin(T[0]), np.nanmax(T[0]), np.nanmean(T[0]))
```

## Status — read this before trusting a number

| | |
|---|---|
| **Works and is tested** | CSQ container decoding (FFF blocks, JPEG-LS payloads), per-frame metadata and timestamps, Planck radiometry, emissivity/atmosphere correction, HDF5/NPY/CSV/MAT export, CLI, PySide6 GUI, 72 automated tests |
| **Validated on a real recording** | A public FLIR **T1020** `.csq` (22 frames, 1024×768, JPEG-LS): payload bytes **sha256-identical** to ExifTool's own extraction, **20/20 metadata fields identical** to ExifTool, temperatures 15.5–33.6 °C matching the scene (outdoor forest, warm animal), **max 0.21 °C** difference from flirpy's independent `raw2temp` |
| **Not yet validated** | **A real FLIR T540 recording.** No T540 file was available. The container is the same T-series FFF/JPEG-LS family, and the code reads dimensions, codec, calibration and timestamps from the file rather than assuming them — but T540 compatibility and absolute accuracy against FLIR Research Studio remain **pending** (`docs/validation.md` lists exactly what to check) |

No numbers in this repository were invented for a camera that was never tested. If your T540
file uses something the decoder cannot read, the program stops with the exact reason instead of
producing plausible-looking temperatures.

## Install (Ubuntu 22.04 / 24.04, x86-64)

```bash
git clone https://github.com/alishavakhi/flir-csq-converter.git
cd flir-csq-converter

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt        # numpy, h5py, imagecodecs, scipy, PySide6

# optional: Qt runtime libraries on a minimal install
sudo apt-get install -y libgl1 libegl1 libxkbcommon-x11-0
# optional: ExifTool as an independent metadata reference
sudo apt-get install -y libimage-exiftool-perl
```

Or `./scripts/setup_ubuntu.sh`, which does the same thing and checks the codecs afterwards.
`python convert_csq.py --demo /tmp/demo.csq` builds a synthetic 464×348 recording so you can try
the whole pipeline before you have a camera file.

## Use

```bash
# graphical interface
python launch_gui.py

# convert with defaults -> experiment_001_temperature.h5 next to the recording
python convert_csq.py experiment_001.csq

# choose outputs and keep the raw counts
python convert_csq.py experiment_001.csq --output results/ --format h5,csv --save-raw --save-metadata

# what is in the file? (no conversion)
python convert_csq.py experiment_001.csq --inspect

# verify an output cube again
python convert_csq.py --verify results/experiment_001_temperature.h5
```

The GUI provides drag-and-drop, recording information (camera, resolution, frames, duration,
payload codec, calibration), output directory and format selection, progress with the current
frame number, cancel, and an "open output folder" button. Conversion runs in a worker thread, so
the window stays responsive.

## What the converter actually decodes

A `.csq` is **not** a video file. It is a flat concatenation of complete FLIR FFF blocks — one
per recorded frame — each holding:

* a **RawData record**: the sensor's 14-bit measurement counts as 16-bit samples, compressed with
  **JPEG-LS** (ITU-T T.87) on every T-series CSQ examined. Uncompressed little/big-endian, PNG
  and lossless-JPEG payloads are handled too;
* a **CameraInfo record**: Planck constants `R1 B F O R2`, emissivity, object distance,
  reflected/atmospheric/window temperatures, relative humidity, atmospheric-transmission
  coefficients, camera model/serial/lens, calibration range and the per-frame **timestamp**
  (Unix seconds + milliseconds + timezone);
* optional records (measurement tools, palette, GPS, audio) that are listed but never used to
  infer temperature.

The frame table is built from each block's own record directory, not by scanning for magic
bytes — `FFF\0` can occur inside compressed pixel data. Corrupt or truncated blocks are reported;
they are never silently dropped or reordered.

## Temperature conversion

Counts → temperature uses the constants stored **in the recording**, never a fixed scale:

```
T[K] = B / ln( R1 / (R2 · (raw + O)) + F )
```

and, when the recording carries the measurement parameters, the counts are first corrected for
emissivity `ε`, reflected apparent temperature, atmospheric transmission `τ`, and an IR window
(`docs/calibration.md` derives every term):

```
c_meas = τ_win · [ ε·τ·c_obj + (1−ε)·τ·c_refl + (1−τ)·c_atm ] + (1−τ_win)·c_win
```

`--mode auto` (default) applies the correction when the file carries the parameters, and falls
back to the Planck inversion with an explicit warning in the JSON report when it does not.
`--mode apparent` forces Planck-only, `--mode object` refuses to run if a parameter is missing.
Pixels outside the calibration range, at the 16-bit rail, or producing a non-physical inversion
become **NaN**, never a clipped number.

## Outputs

| format | file | notes |
|---|---|---|
| HDF5 (default) | `<name>_temperature.h5` | chunked `(1, H, W)`, gzip+shuffle, datasets above |
| NumPy | `<name>_temperature.npy` | written through a memmap, streaming |
| CSV | `<name>_temperature_csv/frame_000123.csv` | one numeric matrix per frame, `%.4f` |
| MATLAB | `<name>_temperature.mat` | `temperature_C` **(frames, H, W)**, `timestamps_s`; needs the cube in RAM |
| report | `<name>_conversion.json` | camera, calibration, parameters used, per-frame statistics, warnings, validation |

`raw_counts` is stored by default (`--no-save-raw` to omit) because it is the lossless evidence
behind every temperature. Nothing is reconstructed or rescaled: it is the array the codec
returned.

## Performance

Frames are decoded, converted and written one at a time; the HDF5 dataset is pre-allocated and
chunked, so memory does not grow with recording length (a test asserts peak RSS stays flat
between 40 and 600 frames). Output is written to `<name>.partial` and renamed only after the last
frame and a successful verification, so a cancelled or crashed run leaves no half cube. Cancel
takes effect at the next frame boundary.

## Tests

```bash
pytest            # 73 tests: decoder, radiometry, exporters, end-to-end, CLI, GUI, memory
```

The suite builds synthetic recordings with **known ground-truth temperatures** — including a
JPEG-LS payload at T540 geometry — and checks that the converted cube returns those temperatures
(worst case 0.004 °C; one uint16 count is ~0.007 °C with these constants). It also checks frame ordering, timestamps,
truncation, garbage recovery, missing calibration, cancellation, reproducibility (bit-identical
re-runs), HDF5 structure and memory behaviour. Synthetic tests prove self-consistency, **not**
camera compatibility; see `docs/validation.md`.

## Documentation

* [`docs/installation.md`](docs/installation.md) — Ubuntu setup, Qt libraries, codec troubleshooting
* [`docs/usage.md`](docs/usage.md) — CLI reference, GUI walkthrough, Python/MATLAB reading
* [`docs/calibration.md`](docs/calibration.md) — equations, parameters, what the numbers mean and do not
* [`docs/format.md`](docs/format.md) — CSQ/FFF/JPEG-LS layout and the sources it was built from
* [`docs/validation.md`](docs/validation.md) — test report, real-recording cross-checks, T540 checklist
* [`examples/`](examples/) — `read_hdf5.py`, `analyze_temperatures.py`, `matlab_read.m`

## Limitations

* Absolute accuracy is bounded by the camera's own calibration and by the measurement parameters
  stored in the file. A wrong emissivity or distance stored on the camera gives a wrong
  temperature, and the report records exactly which values were used (and any override).
* The atmospheric model is FLIR's band-averaged empirical fit, not line-by-line radiative
  transfer; it cannot model a non-uniform path or an unknown window.
* No real T540 recording has been converted yet. Until one is, treat its output as
  container-correct but not accuracy-verified.
* Saturated pixels are NaN, not clipped: a defect hotter than the range is reported as missing,
  not as the range maximum.
* The 8-bit colourised `.mp4`/`.avi` exports some cameras produce contain no radiometry at all
  and are refused on purpose.

## Licence and provenance

MIT (see `LICENSE`). The container and metadata layout follows ExifTool's public
`Image::ExifTool::FLIR` tables and flirpy's independent MIT reader; pixel codecs are third-party
packages (`imagecodecs`/CharLS, `pylibjpeg-libjpeg`, Pillow) under their own licences. No FLIR SDK,
ResearchIR or proprietary library is used, bundled, or required at runtime.
