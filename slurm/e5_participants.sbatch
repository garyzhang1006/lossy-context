#!/usr/bin/env bash
# E5: the participant audit.  Task 0 writes the pooled constrained fits that
# every participant fit pins its nuisances to; tasks 1..E5_SHARDS each fit a
# slice of participants.  Needs LCSA_PARTICIPANTS, the per-participant cloze
# export (columns participant, text_id, word_number, response), which is not
# part of the distributed norms; the job refuses to start without it rather
# than fitting the pooled counts under a participant label.
#
#   sbatch --dependency=afterok:<lcsa-e3-prep id> --array=0-$E5_SHARDS slurm/e5_participants.sbatch
#SBATCH --job-name=lcsa-e5-part
#SBATCH --partition=scu-cpu
#SBATCH --array=0-10
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=12:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%A_%a.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
[ -n "${LCSA_PARTICIPANTS:-}" ] && [ -f "$LCSA_PARTICIPANTS" ] || {
    echo "LCSA_PARTICIPANTS must point at the per-participant cloze file" >&2; exit 2; }
B="$LCSA_ROOT/build"
ARGS=(--kernel "$LCSA_KERNEL" --cache "$B/cache.npz" --out "$LCSA_ART" --e3-out "$LCSA_ART"
      --participants "$LCSA_PARTICIPANTS" --targets "$B/targets.csv" --candidates "$B/candidates.json"
      --n-participants "$N_PART")
[ -z "${LCSA_EXTERNAL_FITS:-}" ] || ARGS+=(--external "$LCSA_EXTERNAL_FITS")
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    lcsa e5 "${ARGS[@]}" --stage pooled
else
    # As above: the count is the registered variable, so requeueing one task
    # alone still computes that task's own range.  Task 0 ran the stage above.
    N="${E5_SHARDS:?set by slurm/env.sh}"
    [ "$SLURM_ARRAY_TASK_ID" -le "$N" ] || {
        echo "array task $SLURM_ARRAY_TASK_ID is outside the registered E5_SHARDS=$N tiling; submit --array=0-$N" >&2
        exit 2; }
    # The participant count is only known after loading, so the shard takes a
    # slice of the sorted participant list by position, N_PART wide at most.
    read -r START STOP <<< "$(shard_range $(( SLURM_ARRAY_TASK_ID - 1 )) "$N" "$N_PART")"
    lcsa e5 "${ARGS[@]}" --stage fits --rep-start "$START" --rep-stop "$STOP"
fi
