#!/usr/bin/env bash
# The CPU legs E1, E2 and E4 on the built cache, one per array task.
#
#   sbatch --dependency=afterok:<lcsa-build id> slurm/estimate.sbatch
#
# E2 and E4 are thousands of independent fits at the registered 200 replicates,
# which the paper budgets at roughly 90 core-hours together with E3; the code
# does not fork, so a task gets four cores for BLAS and a limit under scu-cpu's
# seven-day ceiling.  E3 has its own script because its topic and order nulls
# reload the checkpoint and belong on a GPU.
#SBATCH --job-name=lcsa-est
#SBATCH --partition=scu-cpu
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=3-00:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%A_%a.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"

CACHE="$LCSA_ROOT/build/cache.npz"; TARGETS="$LCSA_ROOT/build/targets.csv"
OUT="$LCSA_ROOT/artifacts"
[ -f "$CACHE" ] || { echo "no cache at $CACHE; run slurm/build.sbatch first" >&2; exit 1; }
N_REP=${N_REP:-200}; N_BOOT=${N_BOOT:-200}     # registered; export smaller for a smoke pass
case "$SLURM_ARRAY_TASK_ID" in
    0) lcsa e1 --cache "$CACHE" --out "$OUT" ;;
    1) lcsa e2 --cache "$CACHE" --out "$OUT" --n-rep "$N_REP" ;;
    2) lcsa e4 --cache "$CACHE" --out "$OUT" --provo-dir "$LCSA_PROVO" \
           --targets "$TARGETS" --subtlex "$LCSA_SUBTLEX" --n-boot "$N_BOOT" ;;
esac
