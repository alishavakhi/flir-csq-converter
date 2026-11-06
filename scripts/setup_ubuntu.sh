#!/usr/bin/env bash
# Ubuntu setup for flir-csq-converter: system packages, virtual environment, codec check.
# Usage: ./scripts/setup_ubuntu.sh [venv-dir]   (default: .venv)
set -euo pipefail

VENV="${1:-.venv}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== system packages =="
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip git \
                          libgl1 libegl1 libxkbcommon-x11-0 libxcb-cursor0
  # optional: independent metadata reference, used by scripts/validate_against_exiftool.py
  sudo apt-get install -y libimage-exiftool-perl || echo "exiftool not installed (optional)"
else
  echo "apt-get not found - install python3, python3-venv and git yourself"
fi

echo "== virtual environment: $VENV =="
python3 -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "== codec check =="
python - <<'PY'
import sys
try:
    import imagecodecs
    ok = hasattr(imagecodecs, "jpegls_decode")
    print(f"imagecodecs {imagecodecs.__version__}  jpegls={ok}")
except Exception as exc:
    print(f"imagecodecs missing: {exc}")
    print("fallback: pip install pylibjpeg pylibjpeg-libjpeg")
    sys.exit(1)
import h5py, numpy
print(f"h5py {h5py.__version__}  numpy {numpy.__version__}")
PY

echo "== demo recording =="
python convert_csq.py --demo /tmp/flir_csq_demo/demo.csq --output /tmp/flir_csq_demo 2>/dev/null || true

cat <<'MSG'

Done. Next steps:
  source .venv/bin/activate
  python launch_gui.py                      # graphical interface
  python convert_csq.py recording.csq       # command line
MSG
