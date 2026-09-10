#!/usr/bin/env bash
# The prefetch is a job now (slurm/prefetch.sbatch): it needs the venv, which
# the login nodes cannot run, and it downloads sequentially so that the Hub
# does not rate-limit the cluster's shared address.  This wrapper submits it.
#
#   bash slurm/prefetch.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
sbatch slurm/prefetch.sbatch
