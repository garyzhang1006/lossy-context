#!/usr/bin/env bash
# Submit build -> {e1, e2, e4} and e3 with dependencies, from the repository root.
#
#   bash slurm/pipeline.sh
#   N_REP=20 N_BOOT=20 bash slurm/pipeline.sh     # smoke pass at twenty replicates
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
jid() { sbatch --parsable --export=ALL "$@" | cut -d';' -f1; }
BUILD=$(jid slurm/build.sbatch);                               echo "build     $BUILD"
EST=$(jid --dependency=afterok:$BUILD slurm/estimate.sbatch);  echo "e1,e2,e4  $EST"
E3=$(jid --dependency=afterok:$BUILD slurm/e3.sbatch);         echo "e3        $E3"
echo "watch with: squeue -u \$USER ; logs under \${LCSA_ROOT:-/athena/accardilab/scratch/\$USER/lossy-context}/logs"
