#!/usr/bin/env bash
# E3 stage one: the constrained fit, the floor calibration and the three
# substantive tilts.  The topic and order tilts reload the checkpoint, so this
# is the one estimation stage on a GPU; everything it writes is read by the
# CPU shards, which never touch the model again.
#
#   sbatch --dependency=afterok:<lcsa-build id> slurm/e3_prepare.sbatch
#SBATCH --job-name=lcsa-e3-prep
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
[ -f "$B/cache.npz" ] || { echo "no cache at $B/cache.npz; run slurm/build.sbatch first" >&2; exit 1; }
require_prefetched "$LCSA_MODEL"
python -c 'import torch; assert torch.cuda.is_available(), "no CUDA device visible"'
lcsa e3 --kernel "$LCSA_KERNEL" --cache "$B/cache.npz" --out "$LCSA_ART" --stage prepare \
    --nulls "${NULLS:-lex,topic,order}" --provo-dir "$LCSA_PROVO" \
    --targets "$B/targets.csv" --candidates "$B/candidates.json" --model "$LCSA_MODEL"
