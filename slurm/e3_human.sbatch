#!/usr/bin/env bash
# E3 stages three and four: task 0 fits the human counts and its precision
# inputs, tasks 1..E3_BOOT_SHARDS each run a shard of the paired contrast
# bootstrap.  Both read only the prepared stage, so they run alongside the
# null replicates; the human fit is written after the nulls only in the sense
# that matters, which is that nothing in it can change a null.
#
#   sbatch --dependency=afterok:<lcsa-e3-prep id> --array=0-$E3_BOOT_SHARDS slurm/e3_human.sbatch
#SBATCH --job-name=lcsa-e3-human
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
CACHE="$LCSA_ROOT/build/cache.npz"; OUT="$LCSA_ROOT/artifacts"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    lcsa e3 --cache "$CACHE" --out "$OUT" --stage human
else
    N=$SLURM_ARRAY_TASK_MAX
    read -r START STOP <<< "$(shard_range $(( SLURM_ARRAY_TASK_ID - 1 )) "$N" "$N_BOOT")"
    lcsa e3 --cache "$CACHE" --out "$OUT" --stage contrast --n-boot "$N_BOOT" \
        --boot-start "$START" --boot-stop "$STOP"
fi
