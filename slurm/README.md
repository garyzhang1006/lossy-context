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
the GPU jobs here ask for twelve hours each, well below that. The preempt
partitions are not used because a build cancelled at hour ten restarts from
zero. The user cap of 250 running jobs is above the largest array here, the
100 tasks of `e3_reps.sbatch`.

All output goes under `/athena/accardilab/scratch/$USER/lossy-context`, on
Lustre and mounted on login and compute nodes alike. The Hugging Face cache
sits next to it in `/athena/accardilab/scratch/$USER/hf` and is shared with the
seed-noise repository. `/scu-storage03/accardilab` is login-only and is never
referenced. The torch wheel carries its own CUDA runtime, so the `cuda/13.0`
module (nvcc only) and apptainer are not needed.

The build pins `gpu:l40s:1`. A single L40S has 46 GB, far more than the 1.5B
model in float16 needs, and pinning keeps the smoke-run throughput figure
comparable to the full build on the same card. Nothing here uses more than one
GPU, so the PCIe-only interconnect is irrelevant. The reference builds and the
E3 prepare stage ask for `gpu:1` of any type because they are short.

## Why shards

Every replicate loop seeds replicate `b` from the run seed and `b` alone, so
the loop can be cut into any number of array tasks without changing a number,
and `lcsa merge` concatenates the shards and refuses a set with a gap or an
overlap. The counts in `env.sh` (`E2_SHARDS=40`, `E3_SHARDS=20`,
`E3_BOOT_SHARDS=10`, `E4_SHARDS=10`) cut the registered 200 replicates into
tasks of ten or twenty, so the whole estimation half runs in the time of its
slowest task rather than the sum. The paper budgets the one-process CPU legs
at about 90 core-hours; spread over roughly 150 concurrent tasks the wall
clock after the builds is set by the ladder and the sweep, a few hours each.

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
   no outbound network, run it as `HF_HUB_OFFLINE=1 bash slurm/pipeline.sh`
   after prefetching `gpt2-large`, `gpt2` and `Qwen/Qwen2.5-0.5B` as well,
   plus the four `LCSA_SWEEP_REFS` checkpoints for `refsweep.sbatch`
   (Llama-3.1-8B is gated and needs an accepted licence on the account).
   The chain includes E6 and the reference sweep; E5 is submitted only when
   `LCSA_PARTICIPANTS` points at the per-participant cloze export, because
   the distributed norms do not carry it.

| script | partition | resources | limit | does |
| --- | --- | --- | --- | --- |
| `build.sbatch` | scu-gpu | 1 L40S, 8 cpu, 48000M | 12 h | selftest, a 20-target smoke build with a throughput extrapolation, then the full build at the registered settings |
| `build_refs.sbatch` | scu-gpu | array 0-2, 1 GPU, 8 cpu, 48000M | 12 h | GPT-2-large, GPT-2-small and Qwen2.5-0.5B caches on the primary's frozen candidate sets |
| `e1.sbatch` | scu-cpu | 4 cpu, 16000M | 12 h | exactness, sensitivity, residual fractions |
| `e2_ladder.sbatch` | scu-cpu | 4 cpu, 16000M | 12 h | the ladder generated under GPT-2-large and fitted under Qwen, plus `e2_theta0.json` |
| `e2_cov.sbatch` | scu-cpu | array of `E2_SHARDS`, 4 cpu, 16000M | 12 h | coverage replicates at rungs 4, 8, 12, 16, 20, 24 and 32; the ceiling is read from these |
| `e3_prepare.sbatch` | scu-gpu | 1 GPU, 8 cpu, 48000M | 12 h | the constrained fit, the N0-PRIME calibration and the lexical, topic and order tilts |
| `e3_reps.sbatch` | scu-cpu | array of readers x `E3_SHARDS`, 4 cpu, 16000M | 12 h | null replicates per reader |
| `e3_human.sbatch` | scu-cpu | array 0-`E3_BOOT_SHARDS`, 4 cpu, 16000M | 12 h | task 0 the human fit, the rest the paired contrast bootstrap |
| `e3_self.sbatch` | scu-cpu | 4 cpu, 16000M | 24 h | the plain floor under the GPT-2-small cache |
| `reliability.sbatch` | scu-cpu | 2 cpu, 8000M | 4 h | G3: debiased split-half JS and participant-half gaze reliability |
| `e5_participants.sbatch` | scu-cpu | array of `E5_SHARDS`+1, 4 cpu, 16000M | 12 h | per-participant half-lives pinned to the pooled nuisances, split-half reliability, alignments; needs `LCSA_PARTICIPANTS` |
| `e6_crossed.sbatch` | scu-cpu | array of `E6_SHARDS`, 4 cpu, 16000M | 12 h | the crossed panel: decay at 4, 8, 16 with and without the N-ORDER tilt |
| `refsweep.sbatch` | scu-gpu | array of 4, 1 gpu, 8 cpu, 64000M | 24 h | Pythia-410M, Pythia-1.4B, Qwen2.5-7B, Llama-3.1-8B caches on the frozen candidates and the human fit under each |
| `confounds.sbatch` | scu-cpu | 4 cpu, 16000M | 12 h | the human counts fitted under each reference cache with its Provo perplexity, plus the Min-K% tertile refits |
| `e4_sweep.sbatch` | scu-cpu | 4 cpu, 32000M | 24 h | the sweep under six references, reading-time gains, hard-window likelihoods |
| `e4_boot.sbatch` | scu-cpu | array of `E4_SHARDS`, 4 cpu, 32000M | 12 h | the argmax bootstrap under the same references |
| `merge.sbatch` | scu-cpu | 2 cpu, 8000M | 1 h | `lcsa merge` and the gate summary |

Every job depends on its inputs with `afterok`, so a failed stage leaves its
dependants pending and `scancel` clears them. `build.sbatch` and
`build_refs.sbatch` skip a build whose `cache.npz` exists, so resubmitting
after a partial run costs only the smoke pass. `N_REP`, `N_BOOT` and the
shard counts are read from the environment, so
`N_REP=20 N_BOOT=20 E2_SHARDS=2 E3_SHARDS=2 E3_BOOT_SHARDS=2 E4_SHARDS=2 bash slurm/pipeline.sh`
exercises every stage at a fraction of the registered cost. A shard that
failed can be resubmitted alone with the same `--rep-start` and `--rep-stop`
and `merge.sbatch` rerun; the replicate seeding guarantees the same rows.

Once the chain has run, `sacct -j <id> --format=JobID,Elapsed,MaxRSS,State`
gives the numbers to tighten `--mem` and `--time` for the next submission.
The limits above are ceilings rather than forecasts.

## The linear-kernel robustness run

`LCSA_KERNEL=linear bash slurm/pipeline.sh robustness` resubmits only the
E3 chain against the caches the registered run built, into
`artifacts_linear`, and `merge.sbatch` writes a manifest of SHA-256 hashes
at the end of either run. `lcsa manifest --out $LCSA_ROOT/artifacts --check`
proves a directory still matches the numbers the paper quotes.
