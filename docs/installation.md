# Installation (Ubuntu Linux, x86-64)

Tested on Ubuntu 24.04 with Python 3.12; the code also runs on 3.9+. No Windows, no Wine, no
FLIR software and no camera connection are needed once you have a `.csq` recording.

## 1. Python environment

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git

git clone https://github.com/alishavakhi/flir-csq-converter.git
cd flir-csq-converter

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs numpy, h5py, imagecodecs, scipy and PySide6. Install the package
itself in editable mode if you want the `flir-csq` / `flir-csq-gui` commands instead of the two
scripts in the repository root:

```bash
pip install -e ".[gui,mat,dev]"
flir-csq --help
```

## 2. Qt runtime libraries

A full Ubuntu desktop already has them. On a minimal image or a container:

```bash
sudo apt-get install -y libgl1 libegl1 libxkbcommon-x11-0 libxcb-cursor0 \
                        libxcb-icccm4 libxcb-keysyms1 libxcb-shape0
```

Symptom if they are missing: `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`.

On a machine without a display (server, CI) the GUI can still be smoke-tested:

```bash
QT_QPA_PLATFORM=offscreen python launch_gui.py --self-test
```

## 3. ExifTool (optional, recommended)

ExifTool is an independent implementation of the same format specification. It is not required
by the converter, but it is the reference used by `scripts/validate_against_exiftool.py`:

```bash
sudo apt-get install -y libimage-exiftool-perl
exiftool -ver
```

## 4. Codec check

The JPEG-LS decoder is what makes T-series CSQ recordings readable. Confirm it:

```bash
python -c "import imagecodecs; print(imagecodecs.__version__, hasattr(imagecodecs,'jpegls_decode'))"
python convert_csq.py --inspect your_recording.csq     # "payload  imagecodecs/CharLS (JPEG-LS)"
```

If `imagecodecs` cannot be installed (very old distributions, unusual CPU):

* install a wheel for your platform: `pip install --only-binary=:all: imagecodecs`
* or `pip install pylibjpeg pylibjpeg-libjpeg` — this covers lossless JPEG but **not** JPEG-LS,
  so T-series CSQ frames will be refused with a clear message rather than mis-decoded.
* last resort for validation only: exiftool stores the same payload per frame
  (`exiftool -b -RawThermalImage file.csq > frame0.bin`) and any JPEG-LS tool can decode it.

## 5. Try it without a recording

```bash
python convert_csq.py --demo /tmp/demo.csq --output /tmp/demo_out
python examples/read_hdf5.py /tmp/demo_out/demo_temperature.h5
```

The demo writes a synthetic 464×348, 12-frame JPEG-LS recording (clearly marked synthetic in the
metadata) and converts it, so you can check the installation end to end.

## 6. Uninstall / clean

The package installs nothing system-wide. Delete the virtual environment and the generated
output files; recordings are never modified (they are opened read-only).

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `cannot decode the jpegls payload ... no JPEG-LS decoder is installed` | `pip install imagecodecs` inside the activated venv |
| `this recording carries no usable Planck calibration` | the file has no radiometric calibration (a colourised export, or a recording made without radiometry). Export raw counts with `--raw-only`; never accept a temperature from such a file |
| `output already exists` | add `--overwrite` (GUI: tick "Overwrite existing outputs") |
| GUI does not start, `xcb` plugin error | install the Qt libraries in §2, or use `QT_QPA_PLATFORM=offscreen` |
| `last frame ... is incomplete` | the camera was still writing when the file was copied. Complete frames are converted; re-copy the file for the rest |
| very slow conversion | check which codec was used (`--inspect`); JPEG-LS decode is the cost. `--stride N` converts every Nth frame |
