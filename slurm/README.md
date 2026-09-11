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

Every GPU job pins `gpu:l40s:1`. What fills a card during a build is the
logits tensor, tokens forwarded times vocabulary size, and that term does not
shrink with the model: sixty candidate rows of a thousand tokens under a
152k-word vocabulary are 18 GB in float16 before any float32 copy, which is
how the seed-noise build ran a 410M model out of memory on a 22 GB Quadro RTX
6000. The reference path now forwards rows in chunks under a token budget
(`lcsa build --max-batch-tokens`, default 16,384) and keeps only the last
position's logits, and `lcsa build` prints the estimate and exits before
loading data if the card cannot hold it. Nothing here uses more than one GPU,
so the PCIe-only interconnect is irrelevant.

## The login nodes cannot run the venv

The login nodes carry CentOS 7 with Python 3.6.8 and no newer python in their
module tree, while every compute node has `/usr/bin/python3` at 3.9.21. The
venv's interpreter is a symlink to that path, so a venv built on a login node
is unusable, and sourcing `env.sh` on a login node used to activate it against
3.6.8 and die on the first `from __future__ import annotations`. Two things
changed. `setup.sh` now submits `setup.sbatch`, which builds the venv on
scu-cpu, and `env.sh` checks the interpreter after activating and exits with
the fix named when it is too old. `pipeline.sh` sources `env.sh` with
`LCSA_VARS_ONLY=1` for the shard counts alone and lets each job activate the
venv on a compute node. Anything that needs `lcsa` runs as a job, including
the checkpoint download.

## Checkpoints are fetched once, then every job runs offline

`prefetch.sbatch` downloads the primary model, the three E4 references and the
four sweep checkpoints one after another in a single scu-cpu job with a retry
on HTTP 429, and writes `$HF_HOME/lcsa_prefetch.done` listing what it fetched.
Once that marker exists `env.sh` sets `HF_HUB_OFFLINE=1` inside every job, and
each GPU script calls `require_prefetched` on its model before loading
anything, so no array task talks to the Hub. Twenty-seven tasks doing so at
once from the cluster's shared address is how the seed-noise run collected
429s. The primary and the three E4 references are fetched first and a failure
there fails the job, because everything waits on it; the four appendix sweep
checkpoints are fetched afterwards and a failure only records the checkpoint as
skipped, so the gated Llama-3.1-8B cannot cancel the build. Each snapshot is
checked for a weights file and refetched allowing `*.bin` when a repository
ships no safetensors, since the alternative is discovering the gap hours later
inside a GPU job running offline. To include the gated checkpoints, accept the
licence on the Hub and run `huggingface-cli login` on a login node (the token
lives in `HF_HOME`) before submitting.

## The registration is a file the merge consumes

The seed-noise audit found a paper whose registered analysis had no
implementation in the released code and whose frozen plan file was not in the
repository. Here `register.sbatch` runs before the build and writes
`$LCSA_ART/registration.json`, holding the design constants imported from the
code (ladder rungs, coverage rungs, panel rungs, the residual-fraction split
procedure, the replicate counts and the shard tiling), the eleven predictions
with their thresholds, the reading rule and the pre-committed gate branches
(G2 failing makes prediction 9 read the analytic ceiling, G4 failing scores
prediction 6 as interval overlap, G5 failing voids predictions 2 to 11 and the
reading rule). Its SHA-256 goes to `registration.sha256` and into the paper.
`merge.sbatch` then runs `lcsa register --check`, which fails if the file was
edited or if any frozen code constant has since drifted, and `lcsa merge`,
which refuses a registered leg whose inputs are absent, refuses a shard tiling
that stops short of the frozen replicate count, and writes `scorecard.csv`
with a status per prediction (supported, falsified, indeterminate, void, not
run) computed from the frozen thresholds. A leg that could not run is recorded
as "not run" through `MERGE_ALLOW_MISSING=e5`, never dropped. An exploratory
directory can still be merged with `lcsa merge --unregistered`, which writes
no scorecard and says so.

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

## What a task's shard range comes from

An array task takes its range from the count in `env.sh` and not from the size
of the array Slurm happens to be running, so requeueing one failed task on its
own (`--array=7`) still computes that task's own replicates; the older form
read `SLURM_ARRAY_TASK_MAX` and silently recomputed the range as if the whole
loop were eight shards wide. A task whose index falls outside the registered
tiling stops with the correct `--array` line in the message rather than dying
on an unbound array element, which is also what a `LCSA_REFS` or
`LCSA_SWEEP_REFS` list of a different length now produces. `pipeline.sh` sizes
both reference arrays from those lists and refuses a shard count above the
replicate count it has to tile, which would otherwise hand the last tasks an
empty range. E5 is the one loop whose length is not known until its file is
read, so the grid is declared as `N_PART` and `lcsa e5` refuses a participant
file with a different count instead of dropping the tail from the last shard.

## Order of operations

1. `bash slurm/setup.sh` (or `sbatch slurm/setup.sbatch`) submits the
   install as a scu-cpu job: the virtualenv on scratch from the compute
   nodes' `/usr/bin/python3`, this checkout with the gpu, rt and dev extras,
   and `lcsa selftest`, which needs no data and no GPU. Wait for it to finish
   before step 4. This job alone writes its log to `lcsa-setup-<jobid>.out`
   in the directory you submitted from, rather than to `$LCSA_ROOT/logs`:
   Slurm fails a job outright when its `--output` directory does not exist,
   and on a fresh account nothing under the lab scratch has been created yet.
   This job creates `$LCSA_ROOT/logs`, so every later job logs there.
2. Copy `Provo_Corpus-Predictability_Norms.csv` and
   `Provo_Corpus-Eyetracking_Data.csv` into
   `/athena/accardilab/scratch/$USER/lossy-context/data/provo/` and
   `SUBTLEXusfrequencyabove1.csv` into `.../lossy-context/data/`. Both
   corpora sit behind browser downloads (OSF and UGent), so no script fetches
   them. Set `LCSA_SUBTLEX` if the file has another name.
3. If any checkpoint is gated (Llama-3.1-8B is), accept its licence and run
   `huggingface-cli login` on a login node so the token sits in `HF_HOME`.
4. `bash slurm/pipeline.sh` submits the chain below, starting with
   `prefetch.sbatch` and `register.sbatch`, which every other job waits on.
   The chain includes E6 and the reference sweep; E5 is submitted and
   registered only when `LCSA_PARTICIPANTS` points at the per-participant
   cloze export, because the distributed norms do not carry it.

| script | partition | resources | limit | does |
| --- | --- | --- | --- | --- |
| `setup.sbatch` | scu-cpu | 4 cpu, 16000M | 2 h | the venv from the compute nodes' python 3.9, the install, `lcsa selftest` |
| `prefetch.sbatch` | scu-cpu | 2 cpu, 8000M | 6 h | every checkpoint downloaded sequentially into `HF_HOME`, then the corpus files checked; writes the offline marker |
| `register.sbatch` | scu-cpu | 1 cpu, 2000M | 10 min | `lcsa register`: the frozen design, predictions and reading rule with their SHA-256 |
| `build.sbatch` | scu-gpu | 1 L40S, 8 cpu, 48000M | 12 h | the memory preflight, a 20-target smoke build with a throughput extrapolation, then the full build at the registered settings |
| `build_refs.sbatch` | scu-gpu | array of `LCSA_REFS`, 1 L40S, 8 cpu, 48000M | 12 h | GPT-2-large, GPT-2-small and Qwen2.5-0.5B caches on the primary's frozen candidate sets |
| `e1.sbatch` | scu-cpu | 4 cpu, 16000M | 12 h | exactness, sensitivity, residual fractions |
| `e2_ladder.sbatch` | scu-cpu | 4 cpu, 16000M | 12 h | the ladder generated under GPT-2-large and fitted under Qwen, plus `e2_theta0.json` |
| `e2_cov.sbatch` | scu-cpu | array of `E2_SHARDS`, 4 cpu, 16000M | 12 h | coverage replicates at rungs 4, 8, 12, 16, 20, 24 and 32; the ceiling is read from these |
| `e3_prepare.sbatch` | scu-gpu | 1 L40S, 8 cpu, 48000M | 12 h | the constrained fit, the N0-PRIME calibration and the lexical, topic and order tilts |
| `e3_reps.sbatch` | scu-cpu | array of readers x `E3_SHARDS`, 4 cpu, 16000M | 12 h | null replicates per reader |
| `e3_human.sbatch` | scu-cpu | array 0-`E3_BOOT_SHARDS`, 4 cpu, 16000M | 12 h | task 0 the human fit, the rest the paired contrast bootstrap |
| `e3_self.sbatch` | scu-cpu | 4 cpu, 16000M | 24 h | the plain floor under the GPT-2-small cache |
| `reliability.sbatch` | scu-cpu | 2 cpu, 8000M | 4 h | G3: debiased split-half JS and participant-half gaze reliability |
| `e5_participants.sbatch` | scu-cpu | array of `E5_SHARDS`+1, 4 cpu, 16000M | 12 h | per-participant half-lives pinned to the pooled nuisances, split-half reliability, alignments; needs `LCSA_PARTICIPANTS` |
| `e6_crossed.sbatch` | scu-cpu | array of `E6_SHARDS`, 4 cpu, 16000M | 12 h | the crossed panel: decay at 4, 8, 16 with and without the N-ORDER tilt |
| `refsweep.sbatch` | scu-gpu | array of `LCSA_SWEEP_REFS`, 1 L40S, 8 cpu, 64000M | 24 h | Pythia-410M, Pythia-1.4B, Qwen2.5-7B, Llama-3.1-8B caches on the frozen candidates and the human fit under each |
| `confounds.sbatch` | scu-cpu | 4 cpu, 16000M | 12 h | the human counts fitted under each reference cache with its Provo perplexity, plus the Min-K% tertile refits |
| `e4_sweep.sbatch` | scu-cpu | 4 cpu, 32000M | 24 h | the sweep under six references, reading-time gains, hard-window likelihoods |
| `e4_boot.sbatch` | scu-cpu | array of `E4_SHARDS`, 4 cpu, 32000M | 12 h | the argmax bootstrap under the same references |
| `merge.sbatch` | scu-cpu | 2 cpu, 8000M | 1 h | `lcsa register --check`, `lcsa merge` with the scorecard, `refsweep.csv` collected from the sweep tasks, the gate summary, the manifest |

Every job depends on its inputs with `afterok`, so a failed stage leaves its
dependants pending and `scancel` clears them. The one exception is
`refsweep.sbatch`, which `merge.sbatch` joins with `afterany`, because the
sweep feeds one appendix table and a checkpoint that ran out of memory or was
never fetched should drop out of that table rather than cancel the merge. `build.sbatch` and
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

`LCSA_KERNEL=linear bash slurm/pipeline.sh robustness` registers E3 alone
under the linear kernel and resubmits only the E3 chain against the caches
the registered run built, into `artifacts_linear`, and `merge.sbatch` writes
a manifest of SHA-256 hashes at the end of either run. `lcsa manifest --out $LCSA_ROOT/artifacts --check`
proves a directory still matches the numbers the paper quotes.
