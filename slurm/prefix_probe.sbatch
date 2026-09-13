#!/usr/bin/env bash
# The prefix probe of registered prediction 12, which no other job produces.
#
#   sbatch --dependency=afterok:<lcsa-build id> --gres=gpu:l40s:1 slurm/prefix_probe.sbatch
#
# Prediction 12 asks by how much a bare truncated prefix moves the candidate
# distribution away from the same prefix scored inside its passage, and the
# scorecard reads that number out of e1_summary, which reads it out of the
# json this job writes.  Without the job the prediction is registered and
# unscored, which is the one outcome a registration is supposed to prevent.
#
# It rescores fifty targets and nothing else, so it costs minutes on a card
# rather than the hours the build costs, and it reuses the build's own frozen
# candidate sets rather than choosing new ones.
#SBATCH --job-name=lcsa-prefix-probe
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=02:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%j.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"

require_prefetched "$LCSA_MODEL"
B="$LCSA_ROOT/build"
for f in "$B/targets.csv" "$B/candidates.json"; do
    [ -f "$f" ] || { echo "no $f; run slurm/build.sbatch first" >&2; exit 1; }
done
# The probe is one file, so a finished run is one test, and a requeued task
# costs no card.
if [ -f "$LCSA_ART/prefix_probe.json" ]; then
    echo "$LCSA_ART/prefix_probe.json exists; remove it to redo the probe"; exit 0
fi
python -c 'import torch; assert torch.cuda.is_available(), "no CUDA device visible"'
lcsa --verbose prefix-probe --provo-dir "$LCSA_PROVO" \
    --targets "$B/targets.csv" --candidates "$B/candidates.json" \
    --model "$LCSA_MODEL" --out "$LCSA_ART"
