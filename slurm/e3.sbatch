#!/usr/bin/env bash
# E3: the zero-decay nulls and the human fit.  The topic and order nulls are
# built from the reference model's behaviour on averaged or shuffled context, so
# this leg reloads the checkpoint and runs on a GPU; the fits after it are CPU.
#
#   sbatch --dependency=afterok:<lcsa-build id> slurm/e3.sbatch
#
# 1-12:00:00 is under scu-gpu's two-day ceiling; if a first pass shows the CPU
# fits dominating, split with `--nulls lex` on scu-cpu and `--nulls topic,order`
# here.
#SBATCH --job-name=lcsa-e3
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=1-12:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%j.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"

B="$LCSA_ROOT/build"; OUT="$LCSA_ROOT/artifacts"
[ -f "$B/cache.npz" ] || { echo "no cache at $B/cache.npz; run slurm/build.sbatch first" >&2; exit 1; }
N_REP=${N_REP:-200}; N_BOOT=${N_BOOT:-200}
python -c 'import torch; assert torch.cuda.is_available(), "no CUDA device visible"'
lcsa e3 --cache "$B/cache.npz" --out "$OUT" --nulls "${NULLS:-lex,topic,order}" \
    --provo-dir "$LCSA_PROVO" --targets "$B/targets.csv" --candidates "$B/candidates.json" \
    --model "$LCSA_MODEL" --n-rep "$N_REP" --n-boot "$N_BOOT"
cat "$OUT/e3_rejection_rates.csv"
