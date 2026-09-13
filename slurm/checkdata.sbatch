#!/usr/bin/env bash
# The data gate, on a cpu node, at the head of the chain.
#
#   sbatch slurm/checkdata.sbatch
#
# `lcsa build` runs G0 too, but it runs it after Slurm has granted an L40S, so
# a corpus the loader cannot reconcile costs a GPU allocation before it says
# why.  Everything the gate measures comes off the two csv files and needs no
# model, so it runs here in seconds and build.sbatch waits on it.  This is a
# job rather than a line in preflight.sh because the login nodes carry Python
# 3.6.8 and cannot import the package at all.
#
# The check also counts the targets the build will admit and the short-context
# targets the sub-cache leg will enumerate, and fails when the second count
# disagrees with the frozen registration, which is the other way this chain
# dies three jobs in.
#SBATCH --job-name=lcsa-check-data
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=00:20:00
#SBATCH --output=/athena/accardilab/scratch/%u/lossy-context/logs/%x-%j.out
set -euo pipefail
ENV="${LCSA_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
[ -f "$ENV" ] || { echo "submit from the repository root or export LCSA_REPO" >&2; exit 2; }
. "$ENV"
for f in "$LCSA_PROVO/Provo_Corpus-Predictability_Norms.csv" "$LCSA_SUBTLEX"; do
    [ -f "$f" ] || { echo "missing $f; see slurm/prefetch.sbatch" >&2; exit 1; }
done
lcsa --verbose check-data --provo-dir "$LCSA_PROVO" --subtlex "$LCSA_SUBTLEX" \
    --out "$LCSA_ROOT/build/checkdata"
