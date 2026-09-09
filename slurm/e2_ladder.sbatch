#!/usr/bin/env bash
# E2 stage one: the recovery ladder, generated under GPT-2-large and fitted under
# the primary, plus the nuisance vector every coverage shard reads.
#
#   sbatch --dependency=afterok:<lcsa-refs id> slurm/e2_ladder.sbatch
#SBATCH --job-name=lcsa-e2-ladder
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=12:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%j.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
CACHE="$LCSA_ROOT/build/cache.npz"; GEN="$(ref_dir "$E2_GEN_REF")/cache.npz"
[ -f "$CACHE" ] || { echo "no cache at $CACHE; run slurm/build.sbatch first" >&2; exit 1; }
[ -f "$GEN" ] || { echo "no generating cache at $GEN; run slurm/build_refs.sbatch first" >&2; exit 1; }
lcsa e2 --cache "$CACHE" --gen-cache "$GEN" --out "$LCSA_ROOT/artifacts" --stage ladder
