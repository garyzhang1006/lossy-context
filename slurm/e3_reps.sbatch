#!/usr/bin/env bash
# E3 stage two: the null replicates, one array task per (reader, shard).
#
#   sbatch --dependency=afterok:<lcsa-e3-prep id> \
#       --array=0-$(( ${#READERS[@]} * E3_SHARDS - 1 )) slurm/e3_reps.sbatch
#
# Task i replicates reader i / E3_SHARDS on shard i % E3_SHARDS.  N0-PRIME is
# regenerated per replicate rather than resampled, so it is the slowest reader
# and sets the limit.
#SBATCH --job-name=lcsa-e3-reps
#SBATCH --partition=scu-cpu
#SBATCH --array=0-99
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=12:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%A_%a.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
read -ra READERS <<< "$E3_READERS"
# The array is ${#READERS[@]} * E3_SHARDS wide; a task past that has no reader.
[ "$SLURM_ARRAY_TASK_ID" -lt $(( ${#READERS[@]} * E3_SHARDS )) ] || {
    echo "task $SLURM_ARRAY_TASK_ID is outside ${#READERS[@]} readers x E3_SHARDS=$E3_SHARDS; submit --array=0-$(( ${#READERS[@]} * E3_SHARDS - 1 ))" >&2
    exit 2; }
READER="${READERS[$(( SLURM_ARRAY_TASK_ID / E3_SHARDS ))]}"
read -r START STOP <<< "$(shard_range $(( SLURM_ARRAY_TASK_ID % E3_SHARDS )) "$E3_SHARDS" "$N_REP")"
lcsa e3 --kernel "$LCSA_KERNEL" --cache "$LCSA_ROOT/build/cache.npz" --out "$LCSA_ART" \
    --stage replicates --readers "$READER" --n-rep "$N_REP" \
    --rep-start "$START" --rep-stop "$STOP"
