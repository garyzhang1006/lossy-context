# Shared environment for the lossy-context Slurm jobs on the SCU cluster.
# Sourced by every sbatch script and by setup.sh; nothing here submits a job.
#
# All output lives under ROOT on the Lustre scratch, mounted on login and compute
# nodes alike.  /scu-storage03 is login-only and is never referenced.  The corpus
# files are expected under $LCSA_DATA, which setup.sh creates and prefetch.sh checks.

export LCSA_ROOT="${LCSA_ROOT:-/athena/accardilab/scratch/$USER/lossy-context}"
export LCSA_VENV="${LCSA_VENV:-$LCSA_ROOT/venv}"
export LCSA_DATA="${LCSA_DATA:-$LCSA_ROOT/data}"
export LCSA_PROVO="${LCSA_PROVO:-$LCSA_DATA/provo}"
export LCSA_SUBTLEX="${LCSA_SUBTLEX:-$LCSA_DATA/SUBTLEXusfrequencyabove1.csv}"
export LCSA_MODEL="${LCSA_MODEL:-Qwen/Qwen2.5-1.5B}"

# One Hugging Face cache for both papers, on scratch rather than the NFS home.
export HF_HOME="${HF_HOME:-/athena/accardilab/scratch/$USER/hf}"
# Set to 1 when the compute nodes have no outbound network, after prefetch.sh.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$LCSA_ROOT"/{build,smoke,artifacts,logs} "$LCSA_PROVO" "$HF_HOME"

if [ -f "$LCSA_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$LCSA_VENV/bin/activate"
else
    echo "no virtualenv at $LCSA_VENV; run slurm/setup.sh on a login node first" >&2
    exit 2
fi
