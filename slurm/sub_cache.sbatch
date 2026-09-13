#!/usr/bin/env bash
# The all-subsets cache prediction 10 is scored from: for every target the depth
# cap admits, the candidate distribution under each of its 2^K retention masks.
# The nested cache holds the K + 1 suffixes alone, which is all graded
# truncation needs and none of what independent deletion needs, so this is a
# second GPU pass over a small slice of the corpus rather than a reread.
#
#   sbatch --dependency=afterok:<lcsa-build id> slurm/sub_cache.sbatch
#
# The cost is sum(2^K) forwards over the selected targets, so it is bounded by
# K_MAX and not by the corpus: at the registered cap of 8 a selected target
# costs at most 256 contexts against the 9 the build paid for it, and the
# whole leg is a fraction of the build it follows on the same card.
#SBATCH --job-name=lcsa-sub-cache
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=12:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%j.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
B="$LCSA_ROOT/build"
for f in "$B/cache.npz" "$B/targets.csv" "$B/candidates.json"; do
    [ -f "$f" ] || { echo "no $f; run slurm/build.sbatch first" >&2; exit 1; }
done
require_prefetched "$LCSA_MODEL"
python -c 'import torch; assert torch.cuda.is_available(), "no CUDA device visible"'

# Written into the build directory, not the artifacts one, because it is keyed
# against that directory's targets.csv and is worthless beside another build.
lcsa --verbose sub-cache --provo-dir "$LCSA_PROVO" --model "$LCSA_MODEL" \
    --targets "$B/targets.csv" --candidates "$B/candidates.json" \
    --out "$B" --k-max "${K_MAX:-8}"
ls -la "$B/sub_caches.npz"
