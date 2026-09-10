#!/usr/bin/env bash
# Download every checkpoint the pipeline loads, one after another, as a job.
#
#   sbatch --dependency=afterok:<lcsa-setup id> slurm/prefetch.sbatch
#
# Two lessons from the seed-noise run are built in.  The downloads run in one
# sequential job rather than inside the GPU array tasks, because twenty-seven
# tasks fetching the same repositories at once from one shared cluster address
# were answered with HTTP 429 by the Hub.  And the script is a job rather than a
# login-node command because it needs the venv, whose interpreter the login
# nodes cannot run.  When it finishes it writes a marker listing what it
# fetched; slurm/env.sh then sets HF_HUB_OFFLINE=1 for every later job, so no
# GPU task ever talks to the Hub.  Provo and SUBTLEX-US sit behind browser
# downloads (https://osf.io/sjefs/ and UGent) and are only checked here.
#SBATCH --job-name=lcsa-prefetch
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=06:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%j.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
. "$ENV"
MARKER="$HF_HOME/lcsa_prefetch.done"
rm -f "$MARKER"
python - <<'PY'
import os, time
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError
models = [os.environ["LCSA_MODEL"], *os.environ["LCSA_REFS"].split(), *os.environ["LCSA_SWEEP_REFS"].split()]
done = []
for m in models:
    for attempt in range(6):
        try:
            p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
            print(f"{m} -> {p}", flush=True)
            done.append(m)
            break
        except HfHubHTTPError as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code == 429 and attempt < 5:
                wait = 60 * (attempt + 1)
                print(f"{m}: HTTP 429, waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            if code in (401, 403):
                raise SystemExit(f"{m} is gated: accept its licence on the Hub and run `huggingface-cli login` "
                                 "on a login node (the token lives in HF_HOME), then resubmit") from exc
            raise
open(os.path.join(os.environ["HF_HOME"], "lcsa_prefetch.done"), "w").write("\n".join(done) + "\n")
PY
missing=0
for f in "$LCSA_PROVO/Provo_Corpus-Predictability_Norms.csv" \
         "$LCSA_PROVO/Provo_Corpus-Eyetracking_Data.csv" "$LCSA_SUBTLEX"; do
    if [ -f "$f" ]; then echo "found   $f"; else echo "MISSING $f"; missing=1; fi
done
[ "$missing" = 0 ] || { echo "copy the missing files into place, then submit the pipeline" >&2; exit 1; }
echo "prefetch complete: $(wc -l < "$MARKER") checkpoints in $HF_HOME; later jobs run with HF_HUB_OFFLINE=1"
