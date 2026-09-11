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
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

# Weights, config and every tokenizer file, but not the duplicate .bin copies
# that several of these repositories still ship beside the safetensors.
KEEP = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"]
WEIGHTS = ("*.safetensors", "*.bin")


def fetch(repo, patterns):
    for attempt in range(6):
        try:
            return snapshot_download(repo, allow_patterns=patterns)
        except HfHubHTTPError as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code == 429 and attempt < 5:
                wait = 60 * (attempt + 1)
                print(f"{repo}: HTTP 429, waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            if code in (401, 403):
                raise SystemExit(
                    f"{repo} is gated and this job has "
                    f"{'a token that does not grant access to it' if os.environ.get('HF_TOKEN') else 'no token'}. "
                    "Accept its licence on the repository's Hub page, then put the token where "
                    "slurm/env.sh looks for it: `huggingface-cli login` on a login node, or write it "
                    "to ~/.cache/huggingface/token, and resubmit") from exc
            raise
    raise SystemExit(f"{repo}: six attempts were all answered with HTTP 429")


def get(repo):
    """Download one checkpoint and return its path, or raise SystemExit.

    A repository that ships only ``pytorch_model.bin`` matches none of the
    weight patterns above, so the snapshot completes with the config and the
    tokenizer alone and the failure surfaces hours later inside a GPU job that
    is running with HF_HUB_OFFLINE=1.  The weights are checked here instead.
    """
    path = Path(fetch(repo, KEEP))
    if not any(path.glob(p) for p in WEIGHTS):
        print(f"{repo}: no safetensors in the snapshot, refetching with the .bin weights", flush=True)
        path = Path(fetch(repo, KEEP + ["*.bin"]))
    if not any(path.glob(p) for p in WEIGHTS):
        raise SystemExit(f"{repo} downloaded no weights file into {path}")
    print(f"{repo} -> {path}", flush=True)
    return path


# The primary and the three E4 and ladder references are what the registered
# legs load, so a failure there fails this job and the chain waiting on it.
# The appendix sweep checkpoints are read only by refsweep.sbatch, which calls
# require_prefetched and fails on its own; a gated Llama-3.1-8B must not cancel
# the build, which is what afterok on this job would otherwise do.
# Said once, up front: a gated repository with no token wastes the whole
# download before failing on the last checkpoint in the list.
GATED = ("meta-llama/", "mistralai/", "google/gemma")
required = [os.environ["LCSA_MODEL"], *os.environ["LCSA_REFS"].split()]
optional = [m for m in os.environ["LCSA_SWEEP_REFS"].split() if m not in required]
if not os.environ.get("HF_TOKEN"):
    blocked = [m for m in required + optional if m.startswith(GATED)]
    if blocked:
        print("no HF_TOKEN in this job's environment, so these gated checkpoints will be skipped: "
              + " ".join(blocked), flush=True)
        print("to include them, accept each licence on the Hub and run `huggingface-cli login` on a "
              "login node, then resubmit this job", flush=True)
    if any(m in required for m in blocked):
        raise SystemExit("a gated checkpoint is in LCSA_MODEL or LCSA_REFS, which the registered "
                         "legs depend on; the run cannot proceed without a token")
done, skipped = [], []
for m in required:
    get(m)
    done.append(m)
for m in optional:
    try:
        get(m)
        done.append(m)
    except SystemExit as exc:
        print(f"SKIPPED {m}: {exc}", flush=True)
        skipped.append(m)
Path(os.environ["HF_HOME"], "lcsa_prefetch.done").write_text("".join(m + "\n" for m in done))
if skipped:
    print("the reference sweep cannot run for: " + " ".join(skipped), flush=True)
    print("fix the access, resubmit slurm/prefetch.sbatch, then slurm/refsweep.sbatch", flush=True)
PY
missing=0
for f in "$LCSA_PROVO/Provo_Corpus-Predictability_Norms.csv" \
         "$LCSA_PROVO/Provo_Corpus-Eyetracking_Data.csv" "$LCSA_SUBTLEX"; do
    if [ -f "$f" ]; then echo "found   $f"; else echo "MISSING $f"; missing=1; fi
done
[ "$missing" = 0 ] || { echo "copy the missing files into place, then submit the pipeline" >&2; exit 1; }
echo "prefetch complete: $(wc -l < "$MARKER") checkpoints in $HF_HOME; later jobs run with HF_HUB_OFFLINE=1"
