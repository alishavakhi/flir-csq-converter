# Usage

## Command line

```
python convert_csq.py RECORDING.csq [options]
```

With no options: converts every complete frame and writes `RECORDING_temperature.h5` next to the
input, with `raw_counts`, `timestamps`, `timestamps_unix_ms`, `metadata`, and a
`RECORDING_conversion.json` report.

### Options

| option | meaning |
|---|---|
| `-o, --output DIR` | output directory (default: next to the recording) |
| `-f, --format LIST` | `h5` (default), `npy`, `csv`, `mat`; combine with commas: `h5,csv` |
| `--save-raw` / `--no-save-raw` | store `raw_counts` uint16 (default on) |
| `--save-mask` | store `valid_mask` uint8 per frame |
| `--save-metadata` / `--no-save-metadata` | write the JSON conversion report (default on) |
| `--mode {auto,apparent,object}` | `auto` corrects for emissivity/atmosphere when the file carries the parameters; `apparent` = Planck inversion only; `object` refuses if a parameter is missing |
| `--raw-only` | export counts and metadata only (for recordings without calibration) |
| `--start N`, `--stop M`, `--stride K` | frame subset; indices are the recording's own frame numbers and are recorded in the report |
| `--overwrite` | replace existing outputs |
| `--compression-level 0-9` | HDF5 gzip level (default 4; 0 = uncompressed, fastest) |
| `--emissivity`, `--distance`, `--reflected-temperature`, `--atmospheric-temperature`, `--humidity`, `--window-transmission`, `--window-temperature` | override values stored in the recording; the override is written into the report |
| `--inspect` | print what the recording contains, convert nothing |
| `--verify H5` | re-verify an output cube |
| `--demo PATH` | build a synthetic recording and convert it |
| `--json` | machine-readable output on stdout |
| `-q/--quiet`, `-v/--verbose` | less / more logging |

Exit codes: `0` success, `2` handled error, `3` temperature conversion unsupported (missing
calibration) or verification failed, `130` cancelled/interrupted.

### Examples

```bash
# inspect first - tells you the camera, resolution, frames, payload codec, frame rate
python convert_csq.py FLIR0001.csq --inspect

# convert, keep raw counts and a CSV matrix per frame
python convert_csq.py FLIR0001.csq -o results --format h5,csv --save-raw

# a long recording: every 5th frame, into MATLAB and HDF5
python convert_csq.py FLIR0001.csq -o results --format h5,mat --stride 5

# the camera stored the wrong emissivity: override it and keep the rest
python convert_csq.py FLIR0001.csq --emissivity 0.87 --distance 0.5 -o results

# a file with no calibration at all: preserve the counts, refuse the temperature
python convert_csq.py broken.csq --raw-only -o results

# batch
for f in ~/recordings/*.csq; do python convert_csq.py "$f" -o results --quiet; done
```

### Batch mode (a directory of recordings, headless)

```bash
python convert_csq.py --input-dir ~/flir-project/input \
                      --output-dir ~/flir-project/output --format h5
```

| option | meaning |
|---|---|
| `--input-dir DIR` | convert every `.csq`/`.seq` under `DIR` (mutually exclusive with a single file) |
| `--output-dir DIR` | where the cubes, reports, logs and state file go (alias of `--output`) |
| `--recursive` / `--no-recursive` | search sub-directories (default on) |
| `--skip-existing` / `--no-skip-existing` | skip recordings already converted with the same settings (default on) |
| `--overwrite` | convert everything again |
| `--on-error continue\|stop` | keep going after a corrupt recording (default) or stop |
| `--dry-run` | list the plan, write nothing |
| `--flat-output` | do not mirror the input tree in the output directory |
| `--naming suffix\|stem` | `<name>_temperature.h5` (default) or `<name>.h5` |
| `--limit N` | process at most N recordings |
| `--extension .EXT` | extra extension to accept (repeatable) |
| `--log-file PATH` | log location (default `<output>/logs/batch_<timestamp>.log`) |
| `--summary-json PATH` | machine-readable run summary (default `<output>/batch_summary.json`) |
| `--json` | print the summary as JSON on stdout |

Exit codes in batch mode: `0` everything converted or skipped, `4` some recordings failed but
others succeeded, `2` nothing converted. A corrupt file never removes a successful conversion, and
re-running the same command skips finished work. See
[`server_usage.md`](server_usage.md) for the full server workflow (installation without root,
`run_server.sh`, SLURM, monitoring, troubleshooting).

```bash
./run_server.sh ~/flir-project/input ~/flir-project/output       # wrapper: venv + logs + exit code
sbatch run_slurm.sbatch ~/flir-project/input ~/flir-project/output   # SLURM compute node
```

## Graphical interface

```bash
python launch_gui.py
```

1. **Select a `.csq` file** — click the drop zone or drag the file onto it. The recording is read
   immediately (header + metadata, no pixels are converted yet).
2. **Check the recording information** — camera model and serial, thermal resolution, number of
   complete frames, duration and measured frame rate, payload codec, timestamp source, and
   whether the recording is radiometrically calibrated.
3. **Configure the output** — directory, formats (HDF5, NumPy, CSV, MATLAB), raw counts, valid
   mask, metadata report, overwrite, conversion mode and an emissivity override.
4. **Start conversion** — progress bar, current/total frame, live log. The window stays
   responsive; **Cancel** stops at the next frame boundary and deletes the partial file.
5. **Done** — the output paths, the number of frames and the temperature span are shown, the
   output is verified, and **Open output folder** takes you there.

## Reading the output in Python

```python
import h5py, numpy as np

with h5py.File("experiment_001_temperature.h5", "r") as f:
    T = f["temperature_C"]          # (frames, H, W) float32, degC, NaN = invalid pixel
    t = f["timestamps"][...]        # seconds since the first converted frame
    raw = f["raw_counts"][...]      # uint16 sensor counts (if saved)
    meta = f["metadata"][()]        # JSON string: camera, calibration, conversion, warnings

frame = T[0]
print(frame.shape, np.nanmin(frame), np.nanmax(frame), np.nanmean(frame))
print(frame[frame.shape[0] // 2, frame.shape[1] // 2])
```

`examples/read_hdf5.py` is this script with argument handling;
`examples/analyze_temperatures.py` plots a heatmap, one pixel versus time, and the mean frame
temperature.

For frames larger than RAM, read slices: `T[1000:1010]`, or `T[:, y0:y1, x0:x1]`.

## Reading the output in MATLAB

```matlab
T = h5read("experiment_001_temperature.h5", "/temperature_C");  % [W H frames]
T = permute(T, [2 1 3]);                                        % [H W frames]
t = h5read("experiment_001_temperature.h5", "/timestamps");
imagesc(double(T(:, :, 1))); axis image; colormap hot; colorbar
```

HDF5 is the recommended route: MATLAB reads it directly and lazily. The `.mat` export
(`--format mat`) writes `temperature_C` as `[frames H W]` plus `timestamps_s` and a metadata
string, but `scipy.io.savemat` needs the whole cube in memory, so it is unsuitable for very long
recordings.

## Verifying a cube

```bash
python convert_csq.py --verify results/experiment_001_temperature.h5
```

The verifier reopens the file and checks dataset presence, dtype, chunking, timestamp monotonicity
and — for sample frames — that the stored temperature is reproduced from the stored raw counts and
the stored calibration (tolerance 1e-3 °C). It is run automatically after every HDF5 conversion and
its result is stored in the JSON report.
