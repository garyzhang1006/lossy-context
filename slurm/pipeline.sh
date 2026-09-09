#!/usr/bin/env bash
# Submit the whole chain with afterok dependencies, from the repository root.
#
#   bash slurm/pipeline.sh
#   N_REP=20 N_BOOT=20 E2_SHARDS=2 E3_SHARDS=2 E3_BOOT_SHARDS=2 E4_SHARDS=2 bash slurm/pipeline.sh
#
# Shard counts come from slurm/env.sh and override the --array lines in the
# scripts, so the registered run and a smoke pass use the same files.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
. slurm/env.sh
read -ra READERS <<< "$E3_READERS"
jid() { sbatch --parsable --export=ALL "$@" | cut -d';' -f1; }
BUILD=$(jid slurm/build.sbatch);                                     echo "build      $BUILD"
REFS=$(jid --dependency=afterok:$BUILD slurm/build_refs.sbatch);      echo "refs       $REFS"
E1=$(jid --dependency=afterok:$BUILD slurm/e1.sbatch);                echo "e1         $E1"
LAD=$(jid --dependency=afterok:$REFS slurm/e2_ladder.sbatch);         echo "e2 ladder  $LAD"
COV=$(jid --dependency=afterok:$LAD --array=0-$((E2_SHARDS - 1)) slurm/e2_cov.sbatch)
echo "e2 cov     $COV"
PREP=$(jid --dependency=afterok:$BUILD slurm/e3_prepare.sbatch);      echo "e3 prepare $PREP"
REPS=$(jid --dependency=afterok:$PREP --array=0-$(( ${#READERS[@]} * E3_SHARDS - 1 )) slurm/e3_reps.sbatch)
echo "e3 reps    $REPS"
HUM=$(jid --dependency=afterok:$PREP --array=0-$E3_BOOT_SHARDS slurm/e3_human.sbatch)
echo "e3 human   $HUM"
SELF=$(jid --dependency=afterok:$REFS slurm/e3_self.sbatch);          echo "e3 self    $SELF"
SW=$(jid --dependency=afterok:$REFS:$PREP slurm/e4_sweep.sbatch);     echo "e4 sweep   $SW"
BOOT=$(jid --dependency=afterok:$REFS:$PREP --array=0-$((E4_SHARDS - 1)) slurm/e4_boot.sbatch)
echo "e4 boot    $BOOT"
MERGE=$(jid --dependency=afterok:$COV:$REPS:$HUM:$SELF:$SW:$BOOT slurm/merge.sbatch)
echo "merge      $MERGE"
echo "watch with: squeue -u \$USER ; logs under $LCSA_ROOT/logs"
