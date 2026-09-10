#!/usr/bin/env bash
# One-off install, as a job: the virtualenv on scratch, this checkout with the
# gpu, rt and dev extras, and the self-test that needs no data and no GPU.
#
#   sbatch slurm/setup.sbatch
#
# This runs on scu-cpu and not on a login node because the login nodes carry
# CentOS 7 and Python 3.6.8, with no newer python in their module tree, while
# every compute node has /usr/bin/python3 at 3.9.21.  A venv built on a login
# node is a venv whose interpreter is 3.6.8 and whose first import dies on
# `from __future__ import annotations`; the seed-noise run lost a day to it.
# The torch wheel bundles its CUDA runtime, so no cuda module is needed.
#SBATCH --job-name=lcsa-setup
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=02:00:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%j.out
set -euo pipefail
REPO="${LCSA_REPO:-${SLURM_SUBMIT_DIR:-}}"
[ -f "$REPO/pyproject.toml" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
export LCSA_ROOT="${LCSA_ROOT:-/athena/accardilab/scratch/$USER/lossy-context}"
export LCSA_VENV="${LCSA_VENV:-$LCSA_ROOT/venv}"
mkdir -p "$LCSA_ROOT"/{build,smoke,artifacts,logs,data/provo}

PY="${LCSA_PYTHON:-/usr/bin/python3}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 9), sys.version' \
    || { echo "$PY on $(hostname) is older than 3.9; set LCSA_PYTHON to a 3.9+ interpreter" >&2; exit 1; }
echo "building the venv with $("$PY" -c 'import sys; print(sys.version.split()[0])') on $(hostname)"
if [ ! -f "$LCSA_VENV/bin/activate" ]; then
    "$PY" -m venv "$LCSA_VENV"
fi
# shellcheck disable=SC1090
. "$LCSA_VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -e "$REPO[gpu,rt,dev]"
python -c 'import sys, torch, transformers; print(sys.version.split()[0], "torch", torch.__version__, "transformers", transformers.__version__)'
lcsa selftest --out "$LCSA_ROOT/selftest"
echo "installed into $LCSA_VENV; next: put the corpus under $LCSA_ROOT/data, then sbatch slurm/prefetch.sbatch"
