#!/usr/bin/env bash
# E1: exactness identities, sensitivity and residual fractions on the cache.
#
#   sbatch --dependency=afterok:<lcsa-build id> slurm/e1.sbatch
#SBATCH --job-name=lcsa-e1
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=12:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%j.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
CACHE="$LCSA_ROOT/build/cache.npz"
[ -f "$CACHE" ] || { echo "no cache at $CACHE; run slurm/build.sbatch first" >&2; exit 1; }
lcsa e1 --cache "$CACHE" --out "$LCSA_ROOT/artifacts"
