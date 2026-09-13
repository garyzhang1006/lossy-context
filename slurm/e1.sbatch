#!/usr/bin/env bash
# E1: exactness identities, sensitivity and residual fractions on the cache.
#
#   sbatch --dependency=afterok:<lcsa-sub-cache id> slurm/e1.sbatch
#
# The dependency is the subset-cache job and not the build, because the
# bridging report of prediction 10 reads sub_caches.npz and that job is the
# only thing that writes it; the subset job itself waits on the build.
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
SUB="$LCSA_ROOT/build/sub_caches.npz"
[ -f "$SUB" ] || { echo "no subset caches at $SUB; run slurm/sub_cache.sbatch first" >&2; exit 1; }
lcsa e1 --kernel "$LCSA_KERNEL" --cache "$CACHE" --out "$LCSA_ART" --sub-caches "$SUB"
