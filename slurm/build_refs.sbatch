#!/usr/bin/env bash
# The zero-decay reference caches for E4 and the cross-model ladder, one array
# task per checkpoint, on the primary build's frozen candidate sets so that every
# reference has row for row the same candidates as the Qwen2.5-1.5B cache.
#
#   sbatch --dependency=afterok:<lcsa-build id> slurm/build_refs.sbatch
#
# GPT-2-large (774M) is the largest of the three, so a run is at most the primary
# build's cost and the limit stays far under scu-gpu's two-day ceiling.
#SBATCH --job-name=lcsa-refs
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=12:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%A_%a.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"

B="$LCSA_ROOT/build"
[ -f "$B/cache.npz" ] || { echo "no cache at $B/cache.npz; run slurm/build.sbatch first" >&2; exit 1; }
read -ra REFS <<< "$LCSA_REFS"
MODEL="${REFS[$SLURM_ARRAY_TASK_ID]}"
OUT=$(ref_dir "$MODEL")
if [ -f "$OUT/cache.npz" ]; then echo "$OUT/cache.npz exists; remove it to rebuild"; exit 0; fi
python -c 'import torch; assert torch.cuda.is_available(), "no CUDA device visible"'
lcsa --verbose build --provo-dir "$LCSA_PROVO" --subtlex "$LCSA_SUBTLEX" \
    --model "$MODEL" --out "$OUT" --max-depth 32 \
    --candidates "$B/candidates.json" --targets "$B/targets.csv"
cmp "$B/targets.csv" "$OUT/targets.csv"
ls -la "$OUT"
