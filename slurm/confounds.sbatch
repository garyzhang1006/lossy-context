#!/usr/bin/env bash
# The competence confounds: the human counts fitted under each reference cache
# reading full context, with that reference's Provo perplexity beside every
# delta.  These are labelled confounds and are not nulls.
#
#   sbatch --dependency=afterok:<lcsa-refs id> slurm/confounds.sbatch
#SBATCH --job-name=lcsa-confounds
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=12:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%j.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
REFARGS=(); for m in $LCSA_REFS; do REFARGS+=(--reference "${m//\//_}=$(ref_dir "$m")"); done
lcsa confounds "${REFARGS[@]}" --out "$LCSA_ROOT/artifacts"
cat "$LCSA_ROOT/artifacts/confounds.csv"
