#!/usr/bin/env bash
# E1: exactness identities, sensitivity and residual fractions on the cache.
#
#   sbatch --dependency=afterok:<lcsa-sub-cache id> slurm/e1.sbatch
#
# The dependency is the subset-cache job and not the build, because the
# bridging report of prediction 10 reads sub_caches.npz and that job is the
# only thing that writes it; the subset job itself waits on the build.
#SBATCH --job-name=lcsa-e1
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=12:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%j.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
CACHE="$LCSA_ROOT/build/cache.npz"
[ -f "$CACHE" ] || { echo "no cache at $CACHE; run slurm/build.sbatch first" >&2; exit 1; }
SUB="$LCSA_ROOT/build/sub_caches.npz"
[ -f "$SUB" ] || { echo "no subset caches at $SUB; run slurm/sub_cache.sbatch first" >&2; exit 1; }
# The probe of prediction 12 is a separate GPU job and this one depends on it
# with afterany, so the flag goes in only when the file is there.  A probe that
# failed then costs prediction 12 its number and costs the other twelve
# nothing, which is the trade the reference sweep already makes at the merge.
PROBE=()
if [ -f "$LCSA_ART/prefix_probe.json" ]; then PROBE=(--prefix-probe "$LCSA_ART/prefix_probe.json")
else echo "note: no $LCSA_ART/prefix_probe.json, so prediction 12 will be unscored"; fi
lcsa e1 --kernel "$LCSA_KERNEL" --cache "$CACHE" --out "$LCSA_ART" --sub-caches "$SUB" \
    ${PROBE[@]+"${PROBE[@]}"}
