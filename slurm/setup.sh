#!/usr/bin/env bash
# One-off install on a login node: virtualenv on scratch, this checkout with the
# gpu, rt and dev extras, and the self-test that needs no data and no GPU.
#
#   bash slurm/setup.sh
#
# The torch wheel bundles its CUDA runtime, so the cuda/13.0 module (nvcc only,
# compute nodes only) is not needed here or in the jobs.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LCSA_ROOT="${LCSA_ROOT:-/athena/accardilab/scratch/$USER/lossy-context}"
export LCSA_VENV="${LCSA_VENV:-$LCSA_ROOT/venv}"
mkdir -p "$LCSA_ROOT"/{build,smoke,artifacts,logs,data/provo}

if [ ! -f "$LCSA_VENV/bin/activate" ]; then
    python3 -c 'import sys; assert sys.version_info >= (3, 9), sys.version' \
        || { echo "python3 on this node is older than 3.9; load a newer python module first" >&2; exit 1; }
    python3 -m venv "$LCSA_VENV"
fi
# shellcheck disable=SC1090
. "$LCSA_VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -e "$REPO[gpu,rt,dev]"
lcsa selftest --out "$LCSA_ROOT/selftest"
echo "installed into $LCSA_VENV; put the corpus under $LCSA_ROOT/data (see slurm/prefetch.sh)"
