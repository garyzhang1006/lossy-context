#!/usr/bin/env bash
# Build the nested ablation cache: the only GPU leg.
#
#   sbatch --dependency=afterok:<lcsa-prefetch id>:<lcsa-register id> slurm/build.sbatch
#
# The job refuses to start unless slurm/prefetch.sbatch has fetched the model,
# and then runs with HF_HUB_OFFLINE=1 from env.sh.  `lcsa build` prints a
# memory estimate for the logits (tokens times vocabulary, the term that
# actually fills a card) and exits before loading data if the GPU cannot
# hold it; l40s below is what that estimate needs at the registered settings.
#
# The registered build is 61,435 context forward passes through a 1.5B model in
# float16.  It costs about 5.35 GPU-hours on a T4 pair, and an L40S (46 GB,
# compute capability 8.9) is many times faster per device, so one GPU and a few
# hours are enough; the limit below stays far under scu-gpu's two-day ceiling.
# The forward pass is a single-device batch with no collective traffic, so the
# PCIe-only interconnect does not matter.  Pinning l40s keeps the smoke-run
# throughput estimate honest for the full build on the same card.
#SBATCH --job-name=lcsa-build
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=12:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%j.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"

require_prefetched "$LCSA_MODEL"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true
python -c 'import torch; assert torch.cuda.is_available(), "no CUDA device visible"'
for f in "$LCSA_PROVO/Provo_Corpus-Predictability_Norms.csv" "$LCSA_SUBTLEX"; do
    [ -f "$f" ] || { echo "missing $f; see slurm/prefetch.sbatch" >&2; exit 1; }
done

# Twenty targets first: every code path the full build uses, in minutes, and a
# throughput figure the full run is extrapolated from.
lcsa --verbose build --provo-dir "$LCSA_PROVO" --subtlex "$LCSA_SUBTLEX" \
    --model "$LCSA_MODEL" --out "$LCSA_ROOT/smoke" --limit 20
python - "$LCSA_ROOT/smoke/g1.json" <<'PY'
import json, sys
rate = json.load(open(sys.argv[1]))["contexts_per_second"]
print(f"smoke run: {rate:.1f} contexts/s; full build estimate {61435 / rate / 3600:.2f} h")
PY

if [ -f "$LCSA_ROOT/build/cache.npz" ]; then
    echo "$LCSA_ROOT/build/cache.npz exists; remove it to rebuild"; exit 0
fi
lcsa --verbose build --provo-dir "$LCSA_PROVO" --subtlex "$LCSA_SUBTLEX" \
    --model "$LCSA_MODEL" --out "$LCSA_ROOT/build" \
    --max-depth 32 --max-candidates 120 --top-k 50
ls -la "$LCSA_ROOT/build"
