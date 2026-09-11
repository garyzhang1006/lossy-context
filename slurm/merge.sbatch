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
# The merge reads the frozen registration, refuses drift or a missing leg, and
# writes scorecard.csv from the registered thresholds.  MERGE_ALLOW_MISSING
# names legs recorded as "not run" on the scorecard, never silently dropped.
lcsa register --out "$OUT" --check
lcsa merge --out "$OUT" --legs "${MERGE_LEGS:-e2,e3,e4,e6}" --n-participants "$N_PART" \
    ${MERGE_ALLOW_MISSING:+--allow-missing "$MERGE_ALLOW_MISSING"}
cat "$OUT/scorecard.csv"
for f in e2_coverage e3_rejection_rates confounds e6_panel e5_summary; do
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
# The appendix's reference sweep: refsweep.sbatch writes one table per
# checkpoint into $OUT/refsweep/<slug>, and nothing else collected them, so the
# fitted half-life across the four families had no single artifact to quote.
python - "$OUT" "$LCSA_SWEEP_REFS" <<'PY'
import csv, sys
from pathlib import Path
out, wanted = Path(sys.argv[1]), sys.argv[2].split()
rows, header, missing = [], None, []
for repo in wanted:
    p = out / "refsweep" / repo.replace("/", "_") / "confounds.csv"
    if not p.exists():
        missing.append(repo)
        continue
    with p.open() as fh:
        r = list(csv.reader(fh))
    if not r:
        missing.append(repo)
        continue
    header = header or ["reference"] + r[0]
    rows.extend([repo] + line for line in r[1:])
if rows:
    dest = out / "refsweep.csv"
    with dest.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"{dest}: {len(rows)} rows from {len(wanted) - len(missing)} checkpoints")
    print(dest.read_text())
if missing:
    print("reference sweep missing for: " + " ".join(missing))
    print("these are appendix checkpoints; the registered legs do not depend on them")
PY

# The gate table: G7 takes the audited GPU-hours of this account from sacct,
# summed over the GPU partition's jobs since the build started.
# awk's END rule prints 0.00 for empty input, so an absent or empty sacct used
# to be recorded as an audited zero rather than as unknown.  Only a positive
# figure is passed; without it G7 reports that the accounting was unavailable.
GPUH=$(sacct -u "$USER" -X -n -p --partition=scu-gpu --format=ElapsedRaw,AllocTRES 2>/dev/null \
       | awk -F'|' '{ n = 0; if (match($2, /gres\/gpu=[0-9]+/)) n = substr($2, RSTART + 9, RLENGTH - 9); s += $1 * n } END { if (s > 0) printf "%.2f", s / 3600 }' || echo "")
[ -n "$GPUH" ] || echo "sacct reported no GPU seconds for $USER on scu-gpu; G7 runs without an audited figure" >&2
lcsa gates --out "$OUT" --build "$LCSA_ROOT/build" ${GPUH:+--gpu-hours "$GPUH"}
cat "$OUT/gates.csv"
# Freeze the outputs: every number in the paper is checked against these hashes
# with `lcsa manifest --out $OUT --check`.
lcsa manifest --out "$OUT"
