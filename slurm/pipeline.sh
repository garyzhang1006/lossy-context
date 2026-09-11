#!/usr/bin/env bash
# Submit the whole chain with afterok dependencies, from the repository root.
#
#   bash slurm/pipeline.sh
#   N_REP=20 N_BOOT=20 E2_SHARDS=2 E3_SHARDS=2 E3_BOOT_SHARDS=2 E4_SHARDS=2 bash slurm/pipeline.sh
#   LCSA_KERNEL=linear bash slurm/pipeline.sh robustness   # after the registered run
#   LCSA_PARTICIPANTS=/path/to/cloze_by_participant.csv bash slurm/pipeline.sh   # adds E5
#
# Shard counts come from slurm/env.sh and override the --array lines in the
# scripts, so the registered run and a smoke pass use the same files.  The
# robustness form reruns only E3 under the linear kernel against the existing
# caches, into artifacts_linear, which is the Kuribayashi appendix table.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# This runs on a login node, whose python cannot run the venv, so env.sh is
# sourced for its variables alone; every job activates the venv itself.
LCSA_VARS_ONLY=1 . slurm/env.sh
[ -f "$LCSA_VENV/bin/activate" ] || { echo "no venv at $LCSA_VENV; sbatch slurm/setup.sbatch and wait for it first" >&2; exit 2; }
read -ra READERS <<< "$E3_READERS"

# env.sh created $LCSA_ROOT/logs above; Slurm fails a job outright when its
# --output directory is absent.  When LCSA_ROOT has been moved off the default
# the logs follow it rather than going to the path baked into the #SBATCH lines.
SBOPT=()
[ "$LCSA_ROOT" = "/athena/accardilab/scratch/$USER/lossy-context" ] \
    || SBOPT=(--output="$LCSA_ROOT/logs/%x-%A_%a.out")
jid() { sbatch --parsable --export=ALL ${SBOPT[@]+"${SBOPT[@]}"} "$@" | cut -d';' -f1; }

# Every array tiles [0, N) by integer division, so a shard count above the
# replicate count hands the last tasks an empty range and `lcsa` dies on
# "replicate range must satisfy 0 <= start < stop".  Catch it at submit time,
# where one message beats a hundred failed array tasks.
while read -r var total; do
    [ "${!var}" -ge 1 ] && [ "${!var}" -le "$total" ] || {
        echo "$var=${!var} must be between 1 and the $total replicates it tiles" >&2; exit 2; }
done <<EOF
E2_SHARDS $N_REP
E3_SHARDS $N_REP
E6_SHARDS $N_REP
E3_BOOT_SHARDS $N_BOOT
E4_SHARDS $N_BOOT
E5_SHARDS $N_PART
EOF
if [ "${1:-}" = robustness ]; then
    [ "$LCSA_KERNEL" != power ] || { echo "robustness needs LCSA_KERNEL=linear" >&2; exit 2; }
    [ -f "$LCSA_ROOT/build/cache.npz" ] || { echo "no primary cache; run the registered pipeline first" >&2; exit 1; }
    REG=$(REGISTER_LEGS=e3 jid slurm/register.sbatch);                 echo "register   $REG  (into $LCSA_ART)"
    PREP=$(jid --dependency=afterok:$REG slurm/e3_prepare.sbatch);      echo "e3 prepare $PREP"
    REPS=$(jid --dependency=afterok:$PREP --array=0-$(( ${#READERS[@]} * E3_SHARDS - 1 )) slurm/e3_reps.sbatch)
    echo "e3 reps    $REPS"
    HUM=$(jid --dependency=afterok:$PREP --array=0-$E3_BOOT_SHARDS slurm/e3_human.sbatch)
    echo "e3 human   $HUM"
    MERGE=$(MERGE_LEGS=e3 jid --dependency=afterok:$REPS:$HUM slurm/merge.sbatch)
    echo "merge      $MERGE  (into $LCSA_ART)"
    exit 0
fi
# The two jobs every other one waits on: the sequential checkpoint download
# and the frozen registration.  Both are cheap to repeat and neither touches a
# result, so they run at the head of every submission.
PRE=$(jid slurm/prefetch.sbatch);                                    echo "prefetch   $PRE"
REGLEGS=e1,e2,e3,e4,e6
if [ -n "${LCSA_PARTICIPANTS:-}" ] && [ -f "$LCSA_PARTICIPANTS" ]; then REGLEGS=$REGLEGS,e5; fi
REG=$(REGISTER_LEGS=$REGLEGS jid slurm/register.sbatch);              echo "register   $REG  (legs $REGLEGS)"
BUILD=$(jid --dependency=afterok:$PRE:$REG slurm/build.sbatch);       echo "build      $BUILD"
read -ra BREFS <<< "$LCSA_REFS"
REFS=$(jid --dependency=afterok:$BUILD --array=0-$(( ${#BREFS[@]} - 1 )) slurm/build_refs.sbatch)
echo "refs       $REFS  (${#BREFS[@]} checkpoints)"
E1=$(jid --dependency=afterok:$BUILD slurm/e1.sbatch);                echo "e1         $E1"
REL=$(jid --dependency=afterok:$BUILD slurm/reliability.sbatch);      echo "reliability $REL"
LAD=$(jid --dependency=afterok:$REFS slurm/e2_ladder.sbatch);         echo "e2 ladder  $LAD"
COV=$(jid --dependency=afterok:$LAD --array=0-$((E2_SHARDS - 1)) slurm/e2_cov.sbatch)
echo "e2 cov     $COV"
PREP=$(jid --dependency=afterok:$BUILD slurm/e3_prepare.sbatch);      echo "e3 prepare $PREP"
REPS=$(jid --dependency=afterok:$PREP --array=0-$(( ${#READERS[@]} * E3_SHARDS - 1 )) slurm/e3_reps.sbatch)
echo "e3 reps    $REPS"
HUM=$(jid --dependency=afterok:$PREP --array=0-$E3_BOOT_SHARDS slurm/e3_human.sbatch)
echo "e3 human   $HUM"
SELF=$(jid --dependency=afterok:$REFS slurm/e3_self.sbatch);          echo "e3 self    $SELF"
CONF=$(jid --dependency=afterok:$REFS slurm/confounds.sbatch);       echo "confounds  $CONF"
SW=$(jid --dependency=afterok:$REFS:$PREP slurm/e4_sweep.sbatch);     echo "e4 sweep   $SW"
BOOT=$(jid --dependency=afterok:$REFS:$PREP --array=0-$((E4_SHARDS - 1)) slurm/e4_boot.sbatch)
echo "e4 boot    $BOOT"
PANEL=$(jid --dependency=afterok:$PREP:$LAD --array=0-$((E6_SHARDS - 1)) slurm/e6_crossed.sbatch)
echo "e6 panel   $PANEL"
read -ra SWEEPREFS <<< "$LCSA_SWEEP_REFS"
SWEEP=$(jid --dependency=afterok:$BUILD --array=0-$(( ${#SWEEPREFS[@]} - 1 )) slurm/refsweep.sbatch)
echo "refsweep   $SWEEP"
LEGS=e2,e3,e4,e6
# The sweep is joined with afterany and the registered legs with afterok: it
# feeds one appendix table that merge reports as missing when a checkpoint did
# not run, so an out-of-memory 8B task must not cancel the whole merge.
DEPS=$E1:$REL:$COV:$REPS:$HUM:$SELF:$CONF:$SW:$BOOT:$PANEL
# E5 needs the per-participant cloze export, which the distributed norms do not
# carry; without it the leg is skipped here instead of failing the chain.
if [ -n "${LCSA_PARTICIPANTS:-}" ] && [ -f "$LCSA_PARTICIPANTS" ]; then
    PART=$(jid --dependency=afterok:$PREP --array=0-$E5_SHARDS slurm/e5_participants.sbatch)
    echo "e5 partic  $PART"
    LEGS=$LEGS,e5; DEPS=$DEPS:$PART
else
    echo "e5 skipped: set LCSA_PARTICIPANTS to the per-participant cloze file to run it"
fi
MERGE=$(MERGE_LEGS=$LEGS jid --dependency=afterok:$DEPS,afterany:$SWEEP slurm/merge.sbatch)
echo "merge      $MERGE  (legs $LEGS)"
echo "watch with: squeue -u \$USER ; logs under $LCSA_ROOT/logs"
