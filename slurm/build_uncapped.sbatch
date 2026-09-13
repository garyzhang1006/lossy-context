#!/usr/bin/env bash
# The uncapped-depth arm of registered prediction 13.
#
#   sbatch --dependency=afterok:<lcsa-prefetch id>:<lcsa-check-data id> \
#          --gres=gpu:l40s:1 slurm/build_uncapped.sbatch
#
# Prediction 13 asks whether the fitted human half-life moves when the depth
# cap comes off, so it needs the same targets cached a second time with no
# truncation.  `lcsa e3 --stage human` scores it from --uncapped-cache and
# refuses an arm that is not deeper, and slurm/e3_human.sbatch passes the flag
# only when this job has written the cache, so a failure here leaves
# prediction 13 unscored rather than cancelling the E3 leg.
#
# LCSA_UNCAPPED_DEPTH is a ceiling and not a target.  Every target caches
# min(context length, depth) prefixes, so a cap above the longest passage
# costs nothing and removes the guess about how long that passage is; the
# attestation in e3_nulls refuses the arm if any target is still truncated.
# The registered build is 61,435 context forward passes at the cap of 32 and
# this one is nearer 67,000, so it costs about what build.sbatch costs.
#SBATCH --job-name=lcsa-build-uncapped
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

require_prefetched "$LCSA_MODEL"
python -c 'import torch; assert torch.cuda.is_available(), "no CUDA device visible"'
for f in "$LCSA_PROVO/Provo_Corpus-Predictability_Norms.csv" "$LCSA_SUBTLEX"; do
    [ -f "$f" ] || { echo "missing $f; see slurm/prefetch.sbatch" >&2; exit 1; }
done
OUT="$LCSA_ROOT/build_uncapped"
# The gate record is the last file the build writes, so a job that died after
# cache.npz is redone rather than taken as finished.
if [ -f "$OUT/g0_cache.json" ]; then
    echo "$OUT/g0_cache.json exists, so this build finished; remove $OUT to rebuild"; exit 0
fi
# Everything except --max-depth repeats build.sbatch, because the two arms have
# to differ in depth alone for the comparison to mean anything.
lcsa --verbose build --provo-dir "$LCSA_PROVO" --subtlex "$LCSA_SUBTLEX" \
    --model "$LCSA_MODEL" --out "$OUT" \
    --max-depth "${LCSA_UNCAPPED_DEPTH:-128}" --max-candidates 120 --top-k 50
ls -la "$OUT"
