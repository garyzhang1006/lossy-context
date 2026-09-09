#!/usr/bin/env bash
# G3, the day-2 reliability gate: debiased split-half JS of the cloze counts
# against the full-context reference, and the participant-half split of gaze.
#
#   sbatch --dependency=afterok:<lcsa-build id> slurm/reliability.sbatch
#SBATCH --job-name=lcsa-reliability
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=04:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%j.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
CACHE="$LCSA_ROOT/build/cache.npz"
[ -f "$CACHE" ] || { echo "no cache at $CACHE; run slurm/build.sbatch first" >&2; exit 1; }
lcsa reliability --cache "$CACHE" --out "$LCSA_ART" --provo-dir "$LCSA_PROVO"
