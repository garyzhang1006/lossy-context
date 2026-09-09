#!/usr/bin/env bash
# The self-reference certification: the plain floor N0 under the GPT-2-small
# cache, on the same frozen candidate sets, so the floor is shown to hold when
# the reference is the generator's own family and not only under Qwen.
#
#   sbatch --dependency=afterok:<lcsa-refs id> slurm/e3_self.sbatch
#SBATCH --job-name=lcsa-e3-self
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=1-00:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%j.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
CACHE="$(ref_dir "$E3_SELF_REF")/cache.npz"
[ -f "$CACHE" ] || { echo "no cache at $CACHE; run slurm/build_refs.sbatch first" >&2; exit 1; }
lcsa e3 --cache "$CACHE" --out "$LCSA_ROOT/artifacts/self_${E3_SELF_REF//\//_}" \
    --nulls none --readers N0 --no-human --n-rep "$N_REP"
