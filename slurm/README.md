# Running on the SCU Slurm cluster

Scripts for the accardilab allocation (Slurm 25.11.6). Submit from the
repository root, or export `LCSA_REPO` pointing at it, since each job finds
`slurm/env.sh` through `SLURM_SUBMIT_DIR`.

## What the cluster imposes

The account is the default, so no `--account` flag appears anywhere. The two
regular partitions accept different QoS names (`scu-cpu` takes normal and
cpu-limited, `scu-gpu` takes normal and gpu-limited, both reject low), so no
script sets `--qos` and the partition default applies. Without `--mem` a job
gets 8000M and without `--time` it gets the partition maximum, which is seven
days on `scu-cpu`, so every script sets both. `scu-gpu` caps at two days, and
the GPU jobs here ask for twelve hours (build) and thirty-six hours (E3), both
well below that. The preempt partitions are not used because a build cancelled
at hour ten restarts from zero.

All output goes under `/athena/accardilab/scratch/$USER/lossy-context`, on
Lustre and mounted on login and compute nodes alike. The Hugging Face cache
sits next to it in `/athena/accardilab/scratch/$USER/hf` and is shared with the
seed-noise repository. `/scu-storage03/accardilab` is login-only and is never
referenced. The torch wheel carries its own CUDA runtime, so the `cuda/13.0`
module (nvcc only) and apptainer are not needed.

The build pins `gpu:l40s:1`. A single L40S has 46 GB, far more than the 1.5B
model in float16 needs, and pinning keeps the smoke-run throughput figure
comparable to the full build on the same card. Nothing here uses more than one
GPU, so the PCIe-only interconnect is irrelevant. E3 asks for `gpu:1` of any
type because its GPU phase is short relative to the CPU fits that follow.

## Order of operations

1. On a login node, `bash slurm/setup.sh`. This builds the virtualenv on
   scratch, installs this checkout with the gpu, rt and dev extras, and runs
   `lcsa selftest`, which needs no data and no GPU.
2. Copy `Provo_Corpus-Predictability_Norms.csv` and
   `Provo_Corpus-Eyetracking_Data.csv` into
   `/athena/accardilab/scratch/$USER/lossy-context/data/provo/` and
   `SUBTLEXusfrequencyabove1.csv` into `.../lossy-context/data/`. Both
   corpora sit behind browser downloads (OSF and UGent), so no script fetches
   them. Set `LCSA_SUBTLEX` if the file has another name.
3. `bash slurm/prefetch.sh` downloads Qwen/Qwen2.5-1.5B into the shared cache
   and checks that step 2 is complete.
4. `bash slurm/pipeline.sh` submits the chain below. If the compute nodes have
   no outbound network, run it as `HF_HUB_OFFLINE=1 bash slurm/pipeline.sh`.

| script | partition | resources | limit | does |
| --- | --- | --- | --- | --- |
| `build.sbatch` | scu-gpu | 1 L40S, 8 cpu, 48000M | 12 h | selftest, a 20-target smoke build with a throughput extrapolation, then the full build at the registered settings |
| `estimate.sbatch` | scu-cpu | array 0-2, 4 cpu, 32000M | 3 d | E1, E2 and E4 on the cache, one array task each |
| `e3.sbatch` | scu-gpu | 1 GPU, 8 cpu, 48000M | 36 h | the lexical, topic and order nulls and the human fit |

The estimation jobs depend on the build with `afterok`, so a failed build
leaves them pending and `scancel` clears them. `build.sbatch` skips the full
build when `build/cache.npz` already exists, so resubmitting after a partial
run costs only the smoke pass. `N_REP` and `N_BOOT` are read from the
environment by the estimation scripts, so a first pass at
`N_REP=20 N_BOOT=20 bash slurm/pipeline.sh` exercises every leg at a fraction
of the registered cost.

Once the chain has run, `sacct -j <id> --format=JobID,Elapsed,MaxRSS,State`
gives the numbers to tighten `--mem` and `--time` for the next submission.
The 90 core-hour estimate for the CPU fits in the top-level README is a paper
budget at 2.5 TFLOP/s and eager attention on Turing GPUs, and the three-day
limit on `estimate.sbatch` is a ceiling rather than a forecast.
