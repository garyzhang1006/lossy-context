#!/usr/bin/env bash
# E4 stage two: the argmax bootstrap over the same references, in E4_SHARDS tasks.
#
#   sbatch --dependency=afterok:<lcsa-refs id>:<lcsa-e3-prep id> --array=0-$((E4_SHARDS-1)) slurm/e4_boot.sbatch
#SBATCH --job-name=lcsa-e4-boot
#SBATCH --partition=scu-cpu
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=12:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%A_%a.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
B="$LCSA_ROOT/build"; OUT="$LCSA_ART"
REFARGS=(); for m in $LCSA_REFS; do REFARGS+=(--reference "${m//\//_}=$(ref_dir "$m")"); done
N=$(( SLURM_ARRAY_TASK_MAX + 1 ))
read -r START STOP <<< "$(shard_range "$SLURM_ARRAY_TASK_ID" "$N" "$N_BOOT")"
lcsa e4 --kernel "$LCSA_KERNEL" --cache "$B/cache.npz" --out "$OUT" --stage boot --provo-dir "$LCSA_PROVO" \
    --targets "$B/targets.csv" --subtlex "$LCSA_SUBTLEX" "${REFARGS[@]}" --tilted-from "$OUT" \
    --n-boot "$N_BOOT" --boot-start "$START" --boot-stop "$STOP"
