# Shared environment for the lossy-context Slurm jobs on the SCU cluster.
# Sourced by every sbatch script and by setup.sh; nothing here submits a job.
#
# All output lives under ROOT on the Lustre scratch, mounted on login and compute
# nodes alike.  /scu-storage03 is login-only and is never referenced.  The corpus
# files are expected under $LCSA_DATA, which setup.sbatch creates and prefetch.sbatch checks.

export LCSA_ROOT="${LCSA_ROOT:-/athena/accardilab/scratch/$USER/lossy-context}"
export LCSA_VENV="${LCSA_VENV:-$LCSA_ROOT/venv}"
export LCSA_DATA="${LCSA_DATA:-$LCSA_ROOT/data}"
export LCSA_PROVO="${LCSA_PROVO:-$LCSA_DATA/provo}"
export LCSA_SUBTLEX="${LCSA_SUBTLEX:-$LCSA_DATA/SUBTLEXusfrequencyabove1.csv}"
export LCSA_MODEL="${LCSA_MODEL:-Qwen/Qwen2.5-1.5B}"

# One Hugging Face cache for both papers, on scratch rather than the NFS home.
export HF_HOME="${HF_HOME:-/athena/accardilab/scratch/$USER/hf}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$LCSA_ROOT"/{build,smoke,artifacts,logs} "$LCSA_PROVO" "$HF_HOME"

# The venv is activated only where its interpreter works.  Its python is a
# symlink to /usr/bin/python3, which is 3.9.21 on every compute node and 3.6.8
# on the login nodes, so sourcing this file on a login node used to activate a
# venv whose first import died on `from __future__ import annotations`.  The
# submit scripts only need the variables above and set LCSA_VARS_ONLY=1.
if [ "${LCSA_VARS_ONLY:-0}" != 1 ]; then
    if [ ! -f "$LCSA_VENV/bin/activate" ]; then
        echo "no virtualenv at $LCSA_VENV; sbatch slurm/setup.sbatch first" >&2
        exit 2
    fi
    # shellcheck disable=SC1091
    . "$LCSA_VENV/bin/activate"
    # Once slurm/prefetch.sbatch has written its marker every job runs offline,
    # so that no GPU array task talks to the Hub: twenty-seven of them doing so
    # at once from the cluster's shared address is how the seed-noise run
    # collected HTTP 429s.  prefetch.sbatch exports 0 before sourcing this file.
    # The flag is decided inside the job and not on the submit node, whose
    # environment sbatch --export=ALL would otherwise carry in.
    if [ -f "$HF_HOME/lcsa_prefetch.done" ]; then
        export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    else
        export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
    fi
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-$HF_HUB_OFFLINE}"
    if ! python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
        echo "the venv's python resolves to $(python --version 2>&1) on $(hostname); this is a login node." >&2
        echo "lcsa needs 3.9+, which the compute nodes have: run this inside sbatch or srun --partition=scu-cpu." >&2
        exit 2
    fi
fi

# Registered replicate counts and the number of array tasks each loop is cut
# into.  Every replicate is seeded from the run seed and its own index, so the
# cut changes wall clock and nothing else; `lcsa merge` checks that the shards
# tile [0, N) with no gap or overlap.
export N_REP="${N_REP:-200}"; export N_BOOT="${N_BOOT:-200}"
export E2_SHARDS="${E2_SHARDS:-40}"; export E3_SHARDS="${E3_SHARDS:-20}"
export E3_BOOT_SHARDS="${E3_BOOT_SHARDS:-10}"; export E4_SHARDS="${E4_SHARDS:-10}"
export E3_READERS="${E3_READERS:-N0 N0-PRIME N-LEX N-TOPIC N-ORDER}"
# Reference checkpoints for E4 and the ladder generator, built on the primary's
# frozen candidate sets by build_refs.sbatch; slugs replace "/" with "_".
export LCSA_REFS="${LCSA_REFS:-gpt2-large gpt2 Qwen/Qwen2.5-0.5B}"
# The reference sweep of the appendix: four families, 410M to 8B parameters,
# each fitted on the frozen candidate sets; gpt2-large is shared with LCSA_REFS.
export LCSA_SWEEP_REFS="${LCSA_SWEEP_REFS:-EleutherAI/pythia-410m EleutherAI/pythia-1.4b Qwen/Qwen2.5-7B meta-llama/Llama-3.1-8B}"
# E5 and E6 shard counts; E5 needs LCSA_PARTICIPANTS, the per-participant cloze file.
export E5_SHARDS="${E5_SHARDS:-10}"; export E6_SHARDS="${E6_SHARDS:-20}"
export E2_GEN_REF="${E2_GEN_REF:-gpt2-large}"
export E3_SELF_REF="${E3_SELF_REF:-gpt2}"
# Retention kernel.  "power" is the registered run; LCSA_KERNEL=linear reruns
# the same jobs for the Kuribayashi robustness table into artifacts_linear so
# the two never overwrite each other.
export LCSA_KERNEL="${LCSA_KERNEL:-power}"
if [ "$LCSA_KERNEL" = power ]; then export LCSA_ART="$LCSA_ROOT/artifacts"
else export LCSA_ART="$LCSA_ROOT/artifacts_$LCSA_KERNEL"; fi

# shard_range I N TOTAL -> "start stop" for array task I of N over [0, TOTAL).
shard_range() {
    local i=$1 n=$2 total=$3
    local start=$(( i * total / n )) stop=$(( (i + 1) * total / n ))
    echo "$start $stop"
}
ref_dir() { echo "$LCSA_ROOT/build_${1//\//_}"; }
# require_prefetched MODEL... -> exit 1 unless prefetch.sbatch fetched each one.
# GPU jobs call this before loading anything so that a checkpoint missing from
# the cache fails at once with the fix named, instead of twenty tasks each
# opening a connection to the Hub.
require_prefetched() {
    local marker="$HF_HOME/lcsa_prefetch.done" m
    [ -f "$marker" ] || { echo "no $marker; sbatch slurm/prefetch.sbatch before any GPU job" >&2; exit 1; }
    for m in "$@"; do
        grep -qxF "$m" "$marker" || { echo "$m is not in $marker; add it to the env lists and resubmit slurm/prefetch.sbatch" >&2; exit 1; }
    done
}
