#!/usr/bin/env bash
#SBATCH --job-name=flir-csq
#SBATCH --output=flir-csq-%j.out
#SBATCH --error=flir-csq-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
## Edit the two lines below for your cluster, or pass them at submit time:
##   sbatch --partition=<partition> --account=<account> run_slurm.sbatch <input> <output>
##SBATCH --partition=cpu
##SBATCH --account=<your-project>

# Convert a directory of FLIR .csq recordings on a SLURM compute node.
#
#   sbatch run_slurm.sbatch ~/flir-project/input ~/flir-project/output
#   sbatch --export=ALL,FLIR_VENV=$HOME/flir-csq-converter/.venv \
#          run_slurm.sbatch "$SCRATCH/input" "$SCRATCH/output"
#
# Do not run long conversions on a login node if your site's policy forbids it;
# this script is meant to be submitted, not executed directly.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${1:-${FLIR_INPUT_DIR:-$HOME/flir-project/input}}"
OUTPUT_DIR="${2:-${FLIR_OUTPUT_DIR:-$HOME/flir-project/output}}"
FORMATS="${FLIR_FORMATS:-h5}"
VENV="${FLIR_VENV:-$ROOT/.venv}"

mkdir -p "$OUTPUT_DIR/logs"

echo "job        : ${SLURM_JOB_ID:-local}"
echo "node       : $(hostname)"
echo "started    : $(date -Is)"
echo "input      : $INPUT_DIR"
echo "output     : $OUTPUT_DIR"
echo "cpus       : ${SLURM_CPUS_PER_TASK:-n/a}   mem: ${SLURM_MEM_PER_NODE:-n/a} MB"

if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
else
  module load python 2>/dev/null || true
  echo "warning: no venv at $VENV - using $(command -v python)" >&2
fi

cd "$ROOT" || exit 2
srun python convert_csq.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --format "$FORMATS" ${FLIR_EXTRA_ARGS:-}
status=$?

echo "finished   : $(date -Is) (exit $status)"
echo "summary    : $OUTPUT_DIR/batch_summary.json"
exit "$status"
