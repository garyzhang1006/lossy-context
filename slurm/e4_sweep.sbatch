#!/usr/bin/env bash
# E4 stage one: the context-limitation sweep under the primary, the three
# reference caches and the two tilted references from the E3 tilts, plus the
# reading-time gains and the hard-window likelihoods.
#
#   sbatch --dependency=afterok:<lcsa-refs id>:<lcsa-e3-prep id> slurm/e4_sweep.sbatch
#SBATCH --job-name=lcsa-e4-sweep
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=1-00:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%j.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
B="$LCSA_ROOT/build"; OUT="$LCSA_ROOT/artifacts"
REFARGS=(); for m in $LCSA_REFS; do REFARGS+=(--reference "${m//\//_}=$(ref_dir "$m")"); done
lcsa e4 --cache "$B/cache.npz" --out "$OUT" --stage sweep --provo-dir "$LCSA_PROVO" \
    --targets "$B/targets.csv" --subtlex "$LCSA_SUBTLEX" "${REFARGS[@]}" --tilted-from "$OUT"
