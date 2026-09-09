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
OUT="$LCSA_ART"
lcsa merge --out "$OUT" --legs "${MERGE_LEGS:-e2,e3,e4}"
for f in e2_coverage e3_rejection_rates confounds; do
    [ -f "$OUT/$f.csv" ] && cat "$OUT/$f.csv"
done
SELF="$OUT/self_${E3_SELF_REF//\//_}"
[ -f "$SELF/e3_rejection_rates.csv" ] && cat "$SELF/e3_rejection_rates.csv"
python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
def load(n):
    p = out / n
    return json.load(open(p)) if p.exists() else None
e2, e3, e4 = load("e2_summary.json"), load("e3_summary.json"), load("e4_summary.json")
if e2: print("G6 coverage", e2["g6"]["measured"], "ceiling", e2["ceiling"])
if e3:
    print("G5 floors", e3["g5"]["measured"])
    print("G4", e3["human"]["g4"]["measured"] if e3.get("human") else None)
if e4: print("E4 selected k", e4["selected"], "prediction 8", e4["prediction_8"])
PY
# Freeze the outputs: every number in the paper is checked against these hashes
# with `lcsa manifest --out $OUT --check`.
lcsa manifest --out "$OUT"
