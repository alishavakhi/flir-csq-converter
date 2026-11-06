#!/usr/bin/env bash
# Run the headless batch conversion on a server.
#
#   ./run_server.sh [INPUT_DIR] [OUTPUT_DIR] [-- extra convert_csq.py options]
#
# Examples
#   ./run_server.sh
#   ./run_server.sh ~/flir-project/input ~/flir-project/output
#   ./run_server.sh data out --format h5,csv --mode apparent
#   FLIR_VENV=/scratch/me/venv ./run_server.sh data out
#
# Environment
#   FLIR_VENV        virtual environment to activate      (default: ./.venv)
#   FLIR_PYTHON      use this interpreter directly, no venv activation
#                    (for module-loaded or system Pythons on HPC clusters)
#   FLIR_INPUT_DIR   input directory                      (default: ~/flir-project/input)
#   FLIR_OUTPUT_DIR  output directory                     (default: ~/flir-project/output)
#   FLIR_FORMATS     comma separated formats              (default: h5; e.g. h5,mat)
#   FLIR_MAT_VERSION MATLAB container for .mat output      (default: auto = 7.3 for big cubes)
#   FLIR_LIMIT       process only the first N recordings   (default: all)
#
# Exit codes: 0 all good (converted or already done), 2 nothing converted,
#             4 some recordings failed but others were converted,
#             130 interrupted (re-run to continue).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 2

# First two non-option arguments are input and output; everything else is
# passed through to convert_csq.py untouched.
INPUT_DIR="${FLIR_INPUT_DIR:-$HOME/flir-project/input}"
OUTPUT_DIR="${FLIR_OUTPUT_DIR:-$HOME/flir-project/output}"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then INPUT_DIR="$1"; shift; fi
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then OUTPUT_DIR="$1"; shift; fi
EXTRA=("$@")
FORMATS="${FLIR_FORMATS:-h5}"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR" || { echo "cannot create $LOG_DIR"; exit 2; }
RUN_LOG="$LOG_DIR/run_$(date +%Y%m%d-%H%M%S)_$$.log"

VENV="${FLIR_VENV:-$ROOT/.venv}"
if [ -n "${FLIR_PYTHON:-}" ]; then
  PY="$FLIR_PYTHON"
  echo "python     : $PY (FLIR_PYTHON, no venv activation)"
elif [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  PY="python"
elif [ -n "${CONDA_PREFIX:-}" ]; then
  echo "no venv at $VENV - using the active conda environment ($CONDA_PREFIX)"
  PY="python"
else
  echo "error: no virtual environment at $VENV" >&2
  echo "       run ./scripts/install_server.sh first, or set FLIR_VENV / FLIR_PYTHON" >&2
  exit 2
fi

echo "input      : $INPUT_DIR"
echo "output     : $OUTPUT_DIR"
echo "formats    : $FORMATS"
echo "interpreter: $(command -v "$PY" || echo "$PY") ($("$PY" --version 2>&1))"
echo "run log    : $RUN_LOG"
echo

ARGS=(--input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --format "$FORMATS")
if [ -n "${FLIR_LIMIT:-}" ]; then
  ARGS+=(--limit "$FLIR_LIMIT")
fi
case ",$FORMATS," in
  *,mat,*|*,matlab,*)
    if [ -n "${FLIR_MAT_VERSION:-}" ]; then
      ARGS+=(--mat-version "$FLIR_MAT_VERSION")
    fi
    ;;
esac

# stdout/stderr go to the console and to the run log; the program writes its own
# batch_*.log and batch_summary.json into the output directory as well.
"$PY" convert_csq.py "${ARGS[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$RUN_LOG"
status=${PIPESTATUS[0]}

echo
case "$status" in
  0) echo "OK: everything converted or already up to date." ;;
  2) echo "FAILED: nothing could be converted - see $RUN_LOG" ;;
  4) echo "PARTIAL: some recordings failed; the successful ones are usable."
     echo "         Failed files are listed above and recorded in $OUTPUT_DIR/batch_summary.json" ;;
  130) echo "INTERRUPTED: completed files are kept; re-run the same command to continue." ;;
  *) echo "exit code $status - see $RUN_LOG" ;;
esac
echo "log        : $RUN_LOG"
echo "summary    : $OUTPUT_DIR/batch_summary.json"
exit "$status"
