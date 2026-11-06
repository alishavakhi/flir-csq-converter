# Running on a Linux server (headless, no sudo, optional SLURM)

This is the workflow for a university Linux server: upload `.csq` recordings into a folder, run
one command, get one HDF5 temperature cube per recording, plus logs and a run summary. No
graphical interface, no display server, no root access and no camera connection are needed.

The GUI (`launch_gui.py`) is only for a desktop; everything below uses `convert_csq.py` and the
`run_server.sh` / `run_slurm.sbatch` wrappers.

---

## 0. Look at the server first (30 seconds)

```bash
python3 --version                    # need 3.9+ (3.10/3.11/3.12 are ideal)
nproc; free -h; df -h "$HOME"        # cores, memory, free space where outputs will go
command -v sbatch squeue sinfo       # is this a SLURM cluster?
sinfo -o "%P %a %D %c %m" 2>/dev/null | head
ulimit -a | grep -E "open files|file size"
```

Notes that matter:

* **No root?** Everything installs into your home directory (a virtual environment). Nothing needs
  `sudo`.
* **No display?** Nothing in the batch path imports Qt (there is a test that asserts this).
* **SLURM?** Do not run long conversions on a login node if your site's policy forbids it — use
  `run_slurm.sbatch` (§8).
* **Quota?** One T540 native recording (464×348, ~470 KB/frame with raw counts, ~150 KB/frame
  without) becomes roughly 0.65 MB per frame with `temperature_C` + `raw_counts` at gzip 4.
  A 10 000-frame recording is ~6.5 GB. Use `--no-save-raw` to roughly halve that.

Record the numbers you see; they are what the guidance below assumes.

## 1. Install (no sudo)

```bash
git clone https://github.com/alishavakhi/flir-csq-converter.git
cd flir-csq-converter
./scripts/install_server.sh              # creates ./.venv and checks the codec

# or choose the venv location (e.g. scratch with more quota):
./scripts/install_server.sh /scratch/$USER/flir-venv
```

The script

1. prints host information (Python version, cores, memory, free disk, SLURM partitions),
2. creates a virtual environment (default `./.venv`),
3. installs numpy, h5py and a JPEG-LS decoder — `imagecodecs` if a wheel exists, otherwise
   `pylibjpeg` + `pylibjpeg-libjpeg`, which decodes the same frames,
4. installs the package itself with `pip install -e .`,
5. runs a codec probe and fails loudly if no JPEG-LS decoder is available.

It never uses `sudo` and never touches system Python. To use a Python module on the cluster:

```bash
module load python/3.11          # whatever your site calls it
PYTHON_BIN=$(command -v python3) ./scripts/install_server.sh
```

## 2. Upload the recordings

Any transfer method works; the recorder is a single file per acquisition.

```bash
# from your laptop, into the server folder
rsync -av --progress ~/camera/csq/  me@server.uni.edu:~/flir-project/input/
# or a single file
scp experiment_001.csq me@server.uni.edu:~/flir-project/input/
# or with SFTP / the university portal — same result
```

Keep the directory structure you want reflected in the output (`--recursive` is on by default and
the tree is mirrored). Do **not** put the output directory inside the input directory.

## 3. Convert everything (one command)

```bash
./run_server.sh ~/flir-project/input ~/flir-project/output
```

or directly, without the wrapper:

```bash
python convert_csq.py \
  --input-dir ~/flir-project/input \
  --output-dir ~/flir-project/output \
  --format h5
```

Both produce, per recording:

```
~/flir-project/output/
├── experiment_001_temperature.h5      # temperature_C, raw_counts, timestamps, metadata
├── experiment_001_conversion.json     # conversion report for that file
├── campaign_b/
│   └── experiment_002_temperature.h5  # input tree is mirrored
├── batch_summary.json                 # machine-readable run report
├── .flir_csq_state.json               # progress record used to skip finished files
└── logs/
    ├── batch_20261106-110502-123456.log   # per-frame progress, warnings, errors
    └── run_20261106-110502_4242.log       # console transcript of the wrapper
```

The original file name is preserved as the stem (`experiment_001.csq` →
`experiment_001_temperature.h5`). Use `--naming stem` if you want `experiment_001.h5`, and
`--flat-output` to collapse the tree (duplicate names are then moved into a sub-folder named after
their parent).

Useful variants:

```bash
# a first quick test on three recordings
python convert_csq.py --input-dir input --output-dir output --limit 3

# see what would happen, write nothing
python convert_csq.py --input-dir input --output-dir output --dry-run

# CSVs too, and a per-pixel validity mask
python convert_csq.py --input-dir input --output-dir output --format h5,csv --save-mask

# force everything to be converted again
python convert_csq.py --input-dir input --output-dir output --overwrite

# stop at the first broken recording instead of continuing
python convert_csq.py --input-dir input --output-dir output --on-error stop
```

Exit codes: `0` everything converted or already up to date, `4` some files failed but the others
converted (usable), `2` nothing could be converted, `130` interrupted.

## 4. Watch progress

The wrapper tails the log to the console, so it is visible in the SSH session. To follow it from
another terminal:

```bash
tail -f ~/flir-project/output/logs/batch_*.log       # per frame, per file
tail -f ~/flir-project/output/logs/run_*.log         # wrapper transcript
# totals only, refreshed every 30 s
watch -n 30 "python -c \"import json;print(json.load(open('$HOME/flir-project/output/batch_summary.json'))['totals'])\""
```

Inside a `tmux`/`screen` session the conversion survives an SSH disconnect:

```bash
tmux new -s flir
./run_server.sh ~/flir-project/input ~/flir-project/output
# Ctrl-b d to detach, `tmux attach -t flir` to come back
```

Interrupt with `Ctrl-C`: the current frame is finished, the partial `.h5` is deleted, the state of
already-converted files is saved, and re-running the same command continues where it stopped.

## 5. Re-runs, interruptions and corrupted files

Nothing is converted twice by accident:

* a recording is skipped when the state file says it was converted **and** its size/mtime/head/tail
  hash are unchanged **and** the output file is still there with the recorded size;
* a changed recording (re-uploaded, longer, different bytes) is converted again;
* a different setting (`--mode`, `--stride`, formats, overrides) invalidates the skip, because the
  settings are part of the state key;
* a deleted output is rebuilt;
* a corrupt or uncalibrated recording is recorded as `failed` in the state and in
  `batch_summary.json`, the log names the reason, and **every other recording is still converted**;
  re-running retries only the failed ones.

To start from scratch, delete `.flir_csq_state.json` (outputs are then overwritten) or pass
`--overwrite`.

## 6. Read the results on the server

```bash
python examples/read_hdf5.py ~/flir-project/output/experiment_001_temperature.h5
```

```python
import h5py, numpy as np
with h5py.File("/home/me/flir-project/output/experiment_001_temperature.h5") as f:
    T = f["temperature_C"]      # (frames, 348, 464) float32, degC
    t = f["timestamps"][...]    # seconds from the first converted frame
    m = f["metadata"][()]       # JSON: camera, calibration, conversion, warnings
print(T.shape, np.nanmin(T[0]), np.nanmax(T[0]), np.nanmean(T[0]))
```

Export a summary table for all files:

```bash
python - <<'EOF'
import json, pathlib
s = json.loads(pathlib.Path.home().joinpath("flir-project/output/batch_summary.json").read_text())
print(s["totals"])
for f in s["files"]:
    print(f["status"], f["frames"], f["input"], "->", f["output"], f.get("error") or "")
EOF
```

## 7. Copy results back

```bash
# from your laptop
rsync -av --progress me@server.uni.edu:~/flir-project/output/ ./flir-results/
```

Copy the `.h5` files and `batch_summary.json`; the logs are useful for a report and for debugging,
the `.csq` recordings stay where they are.

## 8. SLURM clusters (optional)

`run_slurm.sbatch` is a ready job script. Edit the `#SBATCH` header for your site
(`--partition`, `--account`, `--time`, `--mem`) or pass the options at submit time:

```bash
# inspect - is it SLURM?
sinfo; squeue -u "$USER"

# submit the whole directory
sbatch --partition=<your-partition> --account=<your-account> \
       run_slurm.sbatch ~/flir-project/input ~/flir-project/output

# watch it
squeue -u "$USER"
tail -f flir-csq-<jobid>.out
```

The job script activates the same virtual environment (`FLIR_VENV` overrides the path), runs the
converter with `srun`, and writes its own `flir-csq-<jobid>.out/.err` next to where you submitted,
in addition to the `logs/` directory of the output folder.

Resource guidance for a T540 native recording (464×348, JPEG-LS):

| what | value | why |
|---|---|---|
| memory | 2–4 GB is comfortable | one frame in RAM (≈0.65 MB float32 + counts) plus HDF5 buffers; the test suite asserts peak RSS does not grow with frame count |
| CPUs | 1 is enough | conversion is sequential by design; extra cores do not speed up a single file |
| wall time | measure one file first (`--limit 1`) and multiply | throughput is dominated by JPEG-LS decode |
| parallel files | one job per directory | separate output directories avoid two jobs writing the same state file |

Do not submit hundreds of single-file jobs against the same output directory: the state file is
per output directory. Split the recordings into per-job directories, or run one job over the whole
tree.

## 9. Troubleshooting

| symptom | cause / fix |
|---|---|
| `no virtual environment at ...` | run `./scripts/install_server.sh`, or `FLIR_VENV=/path/to/venv ./run_server.sh` |
| `cannot decode the jpegls payload` | no JPEG-LS decoder in the venv: `source .venv/bin/activate && pip install imagecodecs` (or `pylibjpeg pylibjpeg-libjpeg`) |
| `imagecodecs` build fails | no compiler on the node: `pip install pylibjpeg pylibjpeg-libjpeg` instead; the decoder falls back automatically |
| `no recordings found` | wrong `--input-dir`, wrong extension, or files are still being uploaded; `--extension .raw` can add one |
| `output already exists` in single-file mode | add `--overwrite`; batch mode handles this itself |
| job killed, `Killed` in dmesg | out of memory or wall-time limit: lower `--mem`, check free space, split the input |
| disk quota exceeded | `--no-save-raw` halves the size, or convert to another filesystem |
| conversion is slow | it is decoding JPEG-LS for every frame; measure with `--limit 1`, and use `--stride 2` if every frame is not needed |
| results look like temperatures but are wrong by kelvin | the camera stored a different emissivity/distance; check `conversion.parameters` in the metadata and override with `--emissivity`, `--distance` |

## 10. Data and privacy

* Nothing is uploaded anywhere: the program reads local files and writes local files.
* Do not commit recordings, `.h5` outputs, logs or server names to Git. `.gitignore` already
  excludes `*.csq`, `*.seq`, `*.h5`, `*.mat`, `*.npy`, `batch_summary.json`, `.flir_csq_state.json`
  and `logs/`.
* If the recordings are sensitive, they stay on the server; only copy the derived temperature
  matrices you need, and mind your institution's data-handling rules.
