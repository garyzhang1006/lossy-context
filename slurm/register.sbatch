#!/usr/bin/env bash
# Freeze the registration before any leg runs: the design constants, the thirteen
# predictions with their thresholds, the reading rule and the gate branches go
# to $LCSA_ART/registration.json with a SHA-256 sidecar, and the hash is
# printed so the paper can quote it.  lcsa merge refuses to run without the
# file, refuses if it changed, and generates the scorecard from it.
#
#   sbatch --dependency=afterok:<lcsa-setup id> slurm/register.sbatch
#SBATCH --job-name=lcsa-register
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=00:10:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%j.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
LEGS="${REGISTER_LEGS:-e1,e2,e3,e4,e6}"
if [ -f "$LCSA_ART/registration.json" ]; then
    echo "$LCSA_ART/registration.json exists; verifying instead of rewriting"
    lcsa register --out "$LCSA_ART" --check
    exit 0
fi
# shellcheck disable=SC2086
lcsa register --out "$LCSA_ART" --n-rep "$N_REP" --n-boot "$N_BOOT" --kernel "$LCSA_KERNEL" \
    --model "$LCSA_MODEL" --readers $E3_READERS --references $LCSA_REFS \
    --sweep-references $LCSA_SWEEP_REFS --legs "$LEGS" \
    --shards e2=$E2_SHARDS e3=$E3_SHARDS e3_boot=$E3_BOOT_SHARDS e4=$E4_SHARDS e6=$E6_SHARDS
cat "$LCSA_ART/registration.sha256"
