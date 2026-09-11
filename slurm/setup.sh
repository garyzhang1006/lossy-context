#!/usr/bin/env bash
# Kept as the name people reach for.  The install itself is slurm/setup.sbatch
# and must run as a job: the login nodes' python is 3.6.8 and this package
# needs 3.9, which every compute node has at /usr/bin/python3.  Running the
# old login-node install here built a venv that imported nothing.
#
#   bash slurm/setup.sh        # submits slurm/setup.sbatch and prints the job id
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# For LCSA_ROOT and the directories, including the log directory every later
# job writes into.  Variables only: the login nodes cannot run the venv.
LCSA_VARS_ONLY=1 . slurm/env.sh
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null \
   && [ -z "${SLURM_JOB_ID:-}" ] && [ "${LCSA_SETUP_HERE:-0}" = 1 ]; then
    # A node with a new enough python and an explicit opt-in: install in place.
    LCSA_REPO="$PWD" exec bash slurm/setup.sbatch
fi
echo "python3 here is $(python3 -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo missing) on $(hostname); the venv is built inside a scu-cpu job instead"
sbatch slurm/setup.sbatch
echo "watch it with: squeue -u $USER ; log in $PWD/lcsa-setup-<jobid>.out"
