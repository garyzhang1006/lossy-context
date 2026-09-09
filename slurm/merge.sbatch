#!/usr/bin/env bash
# Combine every shard into the registered tables and print them.
#
#   sbatch --dependency=afterok:<every shard job> slurm/merge.sbatch
#SBATCH --job-name=lcsa-merge
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=01:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%j.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
OUT="$LCSA_ROOT/artifacts"
lcsa merge --out "$OUT" --legs e2,e3,e4
cat "$OUT/e2_coverage.csv" "$OUT/e3_rejection_rates.csv"
SELF="$OUT/self_${E3_SELF_REF//\//_}"
[ -f "$SELF/e3_rejection_rates.csv" ] && cat "$SELF/e3_rejection_rates.csv"
[ -f "$OUT/confounds.csv" ] && cat "$OUT/confounds.csv"
python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
e2 = json.load(open(out / "e2_summary.json")); e3 = json.load(open(out / "e3_summary.json"))
e4 = json.load(open(out / "e4_summary.json"))
print("G6 coverage", e2["g6"]["measured"], "ceiling", e2["ceiling"])
print("G5 floors", e3["g5"]["measured"])
print("G4", e3["human"]["g4"]["measured"] if e3.get("human") else None)
print("E4 selected k", e4["selected"], "prediction 8", e4["prediction_8"])
PY
