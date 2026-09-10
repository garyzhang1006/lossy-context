#!/usr/bin/env bash
# The reference-model sweep: the zero-decay cache of each sweep checkpoint on
# the primary build's frozen candidate sets, then the human counts fitted under
# it, so that the fitted half-life can be read across four reference families
# and two orders of magnitude of size.  One array task per checkpoint.
#
#   sbatch --dependency=afterok:<lcsa-build id> slurm/refsweep.sbatch
#
# Llama-3.1-8B and Qwen2.5-7B in float16 fit an L40S with the 32-depth cache
# batched as in the primary build; budget about 12 GPU-hours each, the
# Pythia pair under two together.  Set LCSA_SWEEP_REFS to change the list.
# The card is pinned: the 8B weights alone are 16 GB, and the logits of one
# forward add tokens times vocabulary on top, which a 22 GB card cannot hold.
#SBATCH --job-name=lcsa-refsweep
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=24:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%A_%a.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
B="$LCSA_ROOT/build"
[ -f "$B/cache.npz" ] || { echo "no cache at $B/cache.npz; run slurm/build.sbatch first" >&2; exit 1; }
read -ra REFS <<< "$LCSA_SWEEP_REFS"
MODEL="${REFS[$SLURM_ARRAY_TASK_ID]}"
OUT=$(ref_dir "$MODEL")
if [ ! -f "$OUT/cache.npz" ]; then
    require_prefetched "$MODEL"
    python -c 'import torch; assert torch.cuda.is_available(), "no CUDA device visible"'
    lcsa --verbose build --provo-dir "$LCSA_PROVO" --subtlex "$LCSA_SUBTLEX" \
        --model "$MODEL" --out "$OUT" --max-depth 32 \
        --candidates "$B/candidates.json" --targets "$B/targets.csv"
    cmp "$B/targets.csv" "$OUT/targets.csv"
fi
# The fit under this reference alone; slurm/confounds.sbatch collects every
# reference into one table once all sweep tasks have finished.
lcsa confounds --kernel "$LCSA_KERNEL" --reference "${MODEL//\//_}=$OUT" \
    --cache "$B/cache.npz" --out "$LCSA_ART/refsweep/${MODEL//\//_}"
