#!/usr/bin/env bash
# Install flir-csq on a shared Linux server WITHOUT root, a GUI or a display.
#
#   ./scripts/install_server.sh [VENV_DIR]
#
# Creates a virtual environment (default: ./.venv), installs the headless
# dependencies, and verifies that a JPEG-LS decoder is available. Nothing is
# installed system-wide and no sudo is used.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

VENV="${1:-${FLIR_VENV:-$ROOT/.venv}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "== host =="
"$PYTHON_BIN" --version || { echo "python3 not found: load a Python module or set PYTHON_BIN"; exit 2; }
echo "cores: $(nproc 2>/dev/null || echo '?')   memory: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo '?')"
echo "disk (home): $(df -h "$HOME" 2>/dev/null | awk 'NR==2{print $4" free"}' || echo '?')"
if command -v sinfo >/dev/null 2>&1; then
  echo "SLURM detected; partitions:"
  sinfo -h -o "%P %a %D nodes %c cpus %m MB" 2>/dev/null | head -8 || true
fi

echo "== virtual environment: $VENV =="
if [ ! -d "$VENV" ]; then
  "$PYTHON_BIN" -m venv "$VENV" || {
    echo "could not create a venv. Ask your administrator for python3-venv, or use"
    echo "  PYTHON_BIN=/path/to/python3.x ./scripts/install_server.sh"
    exit 2
  }
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate" || exit 2
python -m pip install --upgrade pip >/dev/null

echo "== dependencies =="
python -m pip install --no-input numpy h5py || exit 2
# scipy is optional: only the small-cube MATLAB v7 container needs it (v7.3 does not).
python -m pip install --no-input scipy || echo "scipy not installed (MATLAB v7 export disabled; v7.3 works)"

if python -m pip install --no-input "imagecodecs>=2023.1.23"; then
  echo "imagecodecs installed"
else
  echo "imagecodecs failed to install (no wheel / no compiler)."
  echo "Falling back to pylibjpeg-libjpeg, which also decodes JPEG-LS:"
  python -m pip install --no-input pylibjpeg pylibjpeg-libjpeg || {
    echo "No JPEG-LS decoder could be installed. CSQ frames cannot be decoded."
    exit 2
  }
fi

# Install the package itself without pulling the GUI back in.
python -m pip install --no-input -e . || {
  echo "package installation failed"
  exit 2
}

echo "== codec check =="
python - <<'PY'
import sys
import numpy as np

ok = []
try:
    import imagecodecs
    ok.append(f"imagecodecs {imagecodecs.__version__} (jpegls={hasattr(imagecodecs, 'jpegls_decode')})")
except Exception as exc:  # noqa: BLE001
    print(f"imagecodecs: {exc}")

try:
    from libjpeg import decode
    probe = None
    try:
        import imagecodecs as _ic
        probe = _ic.jpegls_encode(np.arange(16, dtype=np.uint16).reshape(4, 4))
    except Exception:
        pass
    if probe is not None:
        shape = np.asarray(decode(probe)).shape
        ok.append(f"pylibjpeg-libjpeg (jpegls probe -> {shape})")
    else:
        ok.append("pylibjpeg-libjpeg present (jpegls probe skipped)")
except Exception as exc:  # noqa: BLE001
    print(f"pylibjpeg-libjpeg: {exc}")

import h5py
print("h5py", h5py.__version__, "| numpy", np.__version__)
if ok:
    print("decoders:", "; ".join(ok))
else:
    print("no JPEG-LS decoder found - .csq frames will be refused")
    sys.exit(1)
PY
status=$?

cat <<MSG

== done (exit $status) ==
Activate the environment in new shells with:
  source "$VENV/bin/activate"

Then:
  python convert_csq.py --input-dir ~/flir-project/input \\
                        --output-dir ~/flir-project/output --format h5

or use the wrapper (handles logging and a per-run report):
  ./run_server.sh ~/flir-project/input ~/flir-project/output
MSG
exit "$status"
