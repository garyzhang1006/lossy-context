# Prompt for the model that runs this on the cluster

Paste everything below the line into the assistant that has shell access to a
Weill Cornell SCU login node. It assumes nothing about this repository beyond
the checkout itself.

---

You are running a psycholinguistics experiment pipeline on the Weill Cornell
SCU Slurm cluster. The code is already written, tested and pushed; your job is
to install it, feed it two corpora, submit the job chain, watch it, and
diagnose whatever fails. Do not modify the analysis code, the registration, or
any `#SBATCH` resource line unless a failure message tells you to, because the
design is pre-registered and a changed threshold invalidates the run.

## What the cluster is

Slurm 25.11.6. The account `accardilab` is the default, so never pass
`--account`. Two partitions matter: `scu-cpu`, the default, capped at seven
days, and `scu-gpu`, capped at two days. Never pass `--qos`, because `scu-cpu`
accepts only `normal` and `cpu-limited` while `scu-gpu` accepts only `normal`
and `gpu-limited`, so one global value is rejected on one of them. Never use
`preempt_cpu` or `preempt_gpu`, where `PreemptMode=CANCEL` throws away a build
at hour ten. QOS `normal` runs 250 of your jobs at once and queues the rest,
which shows up as pending jobs with reason `QOSMaxJobsPerUserLimit` and is not
an error.

GPU nodes come in four types and differ mostly in memory per card: `l40s` has
46068 MiB and four cards per node, `rtx5000` has 32760 MiB, `rtx6000` has
23040 MiB and two cards, and the `a40` figure is unpublished. Every GPU job
here asks for `gpu:l40s:1` through the variable `LCSA_GPU_GRES`, because what
fills a card during a build is the logits tensor, which is tokens times a 152k
vocabulary and does not shrink with the model.

Write everything to `/athena/accardilab/scratch/$USER`, which is Lustre and
mounted on login and compute nodes alike. `/scu-storage03/accardilab` is
visible only on the login nodes, so a job that references it fails. `$TMPDIR`
is node-local and Slurm deletes it when the job ends, so nothing goes there.
The login nodes run Python 3.6.8 and the compute nodes run 3.9.21, and this
package needs 3.9, so the virtualenv is built inside a job and never on a
login node. You do not need the `cuda/13.0` module, because the torch wheel
carries its own runtime.

## What to do

Run `bash slurm/preflight.sh` first and after every step that could change its
answer. It checks the partitions, the GPU type, the scratch paths, the
virtualenv, the three corpus files and the array width against the running-job
cap, prints one line per check, and exits non-zero when something would fail
later. Then:

1. `bash slurm/setup.sh` submits the install as a `scu-cpu` job and prints the
   job id. It builds the virtualenv from the compute nodes' `/usr/bin/python3`,
   installs this checkout with its gpu, rt and dev extras, and runs
   `lcsa selftest`, which needs no data and no GPU. Its log lands in
   `lcsa-setup-<jobid>.out` in the directory you submitted from, because on a
   fresh account the scratch log directory does not exist yet and Slurm fails a
   job whose `--output` directory is missing. Wait for it to finish.
2. Put the corpora in place. `Provo_Corpus-Predictability_Norms.csv` and
   `Provo_Corpus-Eyetracking_Data.csv` go in
   `/athena/accardilab/scratch/$USER/lossy-context/data/provo/`, and
   `SUBTLEXusfrequencyabove1.csv` goes in
   `/athena/accardilab/scratch/$USER/lossy-context/data/`. Both sit behind
   browser downloads, on OSF and at UGent, so no script fetches them; ask the
   user for the files if they are not already there.
3. The Hugging Face token. Seven of the eight checkpoints are ungated. The
   eighth, `meta-llama/Llama-3.1-8B`, is gated and belongs to the appendix
   sweep alone, so without a token the sweep skips it, the merge records it as
   missing in `refsweep.csv`, and every registered result still lands. If the
   user wants that row, they accept the licence on the model's Hub page and
   run `huggingface-cli login` on a login node themselves. Never ask them to
   paste a token to you and never echo one, because a token in a transcript is
   a leaked credential. `slurm/env.sh` reads `HF_TOKEN`, then
   `$HF_HOME/token`, then `~/.cache/huggingface/token`, and treats an empty
   value as absent.
4. `bash slurm/pipeline.sh` submits the whole chain with `afterok`
   dependencies and prints one line per job with its id. It starts with
   `prefetch.sbatch`, which downloads every checkpoint sequentially and writes
   a marker that puts every later job in offline mode, because twenty-seven
   tasks hitting the Hub at once from a shared cluster address collects HTTP
   429s. Before that, run the chain once at smoke size to prove the plumbing:
   `N_REP=20 N_BOOT=20 E2_SHARDS=2 E3_SHARDS=2 E3_BOOT_SHARDS=2 E4_SHARDS=2 E6_SHARDS=2 bash slurm/pipeline.sh`
   exercises every stage for a fraction of the cost. The registered run is the
   same command with no variables set.
5. Watch it with `squeue -u $USER` and read logs under
   `/athena/accardilab/scratch/$USER/lossy-context/logs/`. The run is done when
   `lcsa-merge` completes; it writes `scorecard.csv` and the manifest into
   `/athena/accardilab/scratch/$USER/lossy-context/artifacts/`. Report the
   scorecard to the user verbatim, including any leg marked as not run.

The E5 participant leg needs a per-participant cloze export that the
distributed Provo norms do not carry. Without `LCSA_PARTICIPANTS` pointing at
it, `pipeline.sh` prints `e5 skipped` and the rest of the chain is unaffected.

## Reading failures

Every error message in this repository names its own fix, so quote the message
rather than guessing. The ones worth recognising:

- A task that dies on `replicate range must satisfy 0 <= start < stop` got an
  empty shard, which means a shard count exceeds the replicate count it tiles.
  `pipeline.sh` catches this at submit time; if you see it at run time, someone
  resubmitted an array by hand with the wrong width.
- `shards N and M overlap on replicates [a, b)` at merge time means the leg was
  rerun under a different shard count and stale files are left over. Delete the
  named directory and rerun that leg.
- A GPU job that exits on the memory preflight is telling you the card cannot
  hold the batch. Lower `--max-batch-tokens` or keep the `l40s` pin; do not
  widen the gres to `gpu:1` and hope.
- `no <marker>; sbatch slurm/prefetch.sbatch before any GPU job` means the
  prefetch never completed. Rerun it and read its log, since a 401 or 403 there
  is the gated checkpoint and a 429 is rate limiting that its backoff should
  have absorbed.
- An array pending with `QOSMaxJobsPerUserLimit` is waiting, not broken.
- A failed stage leaves its dependants pending forever, so `scancel` them once
  you have diagnosed the cause, fix it, and resubmit that stage alone. Every
  replicate is seeded from the run seed and its own index, so a resubmitted
  shard reproduces the same rows and the merge is safe to rerun.

`build.sbatch` and `build_refs.sbatch` skip a build whose `cache.npz` already
exists, so resubmitting after a partial run costs only the smoke pass. After
the run, `sacct -j <id> --format=JobID,Elapsed,MaxRSS,State` gives the numbers
to tighten the limits for next time.

Tell the user what failed with the exact message and the job id. Do not
silently retry a stage more than once, and do not edit the registration.
