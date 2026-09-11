#!/usr/bin/env bash
# E6: the crossed panel, real decay at d_half in {4, 8, 16} with the N-ORDER
# mismatch tilted on top at the alpha E3 calibrated, against a plain arm that
# shares its retention draws.  Reads the E3 prepared stage and e2_theta0.json.
#
#   sbatch --dependency=afterok:<lcsa-e3-prep id>:<lcsa-e2-ladder id> \
#       --array=0-$((E6_SHARDS-1)) slurm/e6_crossed.sbatch
#
# At 200 replicates and twenty shards a task fits ten replicates at three rungs
# in two arms under both estimators, 120 profiled fits.
#SBATCH --job-name=lcsa-e6-panel
#SBATCH --partition=scu-cpu
#SBATCH --array=0-19
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=12:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%A_%a.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
# The shard count comes from the registered variable, not from the size of the
# array Slurm happens to be running: requeueing one failed task on its own
# (`--array=7`) leaves SLURM_ARRAY_TASK_MAX at 7, and the old form then
# recomputed that task's range as if the whole loop were eight shards wide.
N="${E6_SHARDS:?set by slurm/env.sh}"
[ "$SLURM_ARRAY_TASK_ID" -lt "$N" ] || {
    echo "array task $SLURM_ARRAY_TASK_ID is outside the registered E6_SHARDS=$N tiling; submit --array=0-$((N - 1))" >&2
    exit 2; }
read -r START STOP <<< "$(shard_range "$SLURM_ARRAY_TASK_ID" "$N" "$N_REP")"
lcsa e6 --kernel "$LCSA_KERNEL" --cache "$LCSA_ROOT/build/cache.npz" --out "$LCSA_ART" \
    --stage panel --n-rep "$N_REP" --rep-start "$START" --rep-stop "$STOP"
