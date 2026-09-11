#!/usr/bin/env bash
# E2 stage two: the coverage replicates, cut into E2_SHARDS array tasks.
#
#   sbatch --dependency=afterok:<lcsa-e2-ladder id> --array=0-$((E2_SHARDS-1)) slurm/e2_cov.sbatch
#
# At the registered 200 replicates and forty shards, a task fits five replicates
# at the seven coverage rungs (4, 8, 12, 16, 20, 24, 32) under both estimators,
# seventy profiled fits, inside the limit with room.
#SBATCH --job-name=lcsa-e2-cov
#SBATCH --partition=scu-cpu
#SBATCH --array=0-39
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=12:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%A_%a.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
CACHE="$LCSA_ROOT/build/cache.npz"; GEN="$(ref_dir "$E2_GEN_REF")/cache.npz"
# The shard count comes from the registered variable, not from the size of the
# array Slurm happens to be running: requeueing one failed task on its own
# (`--array=7`) leaves SLURM_ARRAY_TASK_MAX at 7, and the old form then
# recomputed that task's range as if the whole loop were eight shards wide.
N="${E2_SHARDS:?set by slurm/env.sh}"
[ "$SLURM_ARRAY_TASK_ID" -lt "$N" ] || {
    echo "array task $SLURM_ARRAY_TASK_ID is outside the registered E2_SHARDS=$N tiling; submit --array=0-$((N - 1))" >&2
    exit 2; }
read -r START STOP <<< "$(shard_range "$SLURM_ARRAY_TASK_ID" "$N" "$N_REP")"
lcsa e2 --kernel "$LCSA_KERNEL" --cache "$CACHE" --gen-cache "$GEN" --out "$LCSA_ART" \
    --stage coverage --n-rep "$N_REP" --rep-start "$START" --rep-stop "$STOP"
