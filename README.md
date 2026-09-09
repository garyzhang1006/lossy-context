# lossy-context

Code for *Absorption: what a fitted context-decay parameter actually measures*, an
identifiability audit of the memory-decay parameters that lossy-context surprisal
work fits through a frozen language-model likelihood.

The package is called `lcsa`. It builds a nested ablation cache from the Provo
corpus and a reference language model, fits a one-parameter retention kernel
through two estimators, and runs the four experiments the paper reports. Every
number in the paper comes out of `lcsa` and nothing is hand-transcribed.

## What the code computes

The retention kernel is `r(d; delta) = (1 + d)^(-delta)`, so full retention sits at
`delta = 0` as an interior point of the family rather than at a boundary the nulls
can never reach. Graded random truncation draws one uniform variate per context,
which means the mask distribution has at most `K + 1` atoms and the marginal
likelihood is an exact finite sum over at most 33 cached distributions. There is no
proposal distribution and no importance weight anywhere in the likelihood, so a fit
at a new `delta` is a reweighting of a cache that was computed once.

Two estimators share that likelihood. The naive one tilts the mixture by a lexical
floor and a temperature, giving a nuisance span of dimension two. The repaired one
adds an explicit production channel of four lexical features and a prior-mention
indicator, giving dimension seven. The absorption criterion says the fitted decay
is identified only up to the part of human mismatch that lies outside that span, so
the code reports the residual fraction `||h_perp|| / ||h||` for every reader under
both estimators, and that quantity depends on no fit of `delta` at all.

Inference is a cluster-robust efficient-score test at `delta = 0` with the passage
as the independent unit. Both the CR1 and the CR3 jackknife variance are computed
and the more conservative one becomes the headline, because 55 clusters is few
enough that the choice matters. The naive likelihood ratio is printed beside it for
every reader, since the gap between the two is the point of one of the registered
predictions.

## Install

```bash
pip install -e ".[gpu,rt,dev]"
```

The base install needs only numpy, scipy, pandas, pyyaml and tqdm, and it is enough
to run every estimation leg on a cache somebody else built. The `gpu` extra pulls
torch and transformers, which the build step needs. The `rt` extra pulls statsmodels
for the mixed-effects reading-time models; without it the reading-time leg falls
back to ordinary least squares with passage fixed effects and says so in its output.

## Check the install before spending anything

```bash
lcsa selftest
```

This builds a 120-target synthetic corpus in memory, runs all four experiments on
it, and prints `SELFTEST PASSED`. It needs no data, no GPU and no network, and it
finishes in about a minute. If it fails, nothing downstream is worth starting.

The test suite is the stronger check and takes a couple of minutes:

```bash
pytest -q
```

Every test asserts a mathematical identity or an invariant rather than a
particular implementation. The graded-truncation simulation matches the exact
mixture to within 4e-3 over 400,000 draws, the packed block-diagonal forward pass
matches an unpadded reference within 5e-3 log units on a real transformer, and a
corpus generated at a true window of four words is recovered by both the likelihood
sweep and the AIC hard-window sweep.

## Data

Two files are needed and neither is redistributable here.

Provo comes from OSF at https://osf.io/sjefs/ and the loader wants the two CSVs
under one directory: `Provo_Corpus-Predictability_Norms.csv` and
`Provo_Corpus-Eyetracking_Data.csv`. The predictability norms are Latin-1 encoded
and contain the literal response string `NA`, both of which a default pandas read
destroys silently, so use the loader rather than `read_csv`.

SUBTLEX-US supplies the unigram channel and comes from
https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus.
Pass it with `--subtlex`. Without it the code substitutes a uniform distribution,
labels every downstream result with that fact, and logs a warning, so a missing
frequency file degrades the result instead of crashing the run.

## Running the pipeline

The pipeline splits at one seam. Building the nested cache needs a GPU and takes
hours; everything after it is CPU-only and reads that cache from disk. Each command
writes its own artifacts and prints a short summary, so a session that dies halfway
loses at most the leg it was running.

```bash
lcsa --config config/default.yaml build --provo-dir data/provo --out artifacts/build
lcsa e1 --cache artifacts/build/cache.npz --out artifacts
lcsa e2 --cache artifacts/build/cache.npz --out artifacts --n-rep 200
lcsa e3 --cache artifacts/build/cache.npz --out artifacts --nulls lex,topic,order \
    --provo-dir data/provo
lcsa e4 --cache artifacts/build/cache.npz --out artifacts --provo-dir data/provo
```

The build step runs the data-integrity gate before it touches the GPU and refuses
to continue if the gate fails, because a decode error or a misaligned word index
would propagate into every cached distribution with nothing downstream to catch it.
Pass `--force` to override that refusal, which is worth doing only when you have
read the gate record and understand which check fired.

`e3` needs the reference checkpoint again for the topic and order nulls, since both
are built from the reference model's own behaviour on shuffled or averaged context.
The lexical null needs only the cache, so `--nulls lex` runs on CPU alone.

### Stages and shards

Every replicate loop seeds replicate `b` from the run seed and `b` alone, so a
range of replicates computed in one process is the same numbers whether or not
the others ran alongside it. Each leg therefore splits into a stage that runs
once and shards that any number of processes can compute, and `lcsa merge`
concatenates the shards and applies the same summariser the one-process run
uses. The test suite checks that the two paths write byte-identical artifacts.

```bash
lcsa e2 --cache C --gen-cache R --out A --stage ladder
lcsa e2 --cache C --gen-cache R --out A --stage coverage --n-rep 200 --rep-start 0 --rep-stop 10
lcsa e3 --cache C --out A --stage prepare --nulls lex,topic,order --provo-dir data/provo
lcsa e3 --cache C --out A --stage replicates --readers N0-PRIME --n-rep 200 --rep-start 0 --rep-stop 10
lcsa e3 --cache C --out A --stage human
lcsa e3 --cache C --out A --stage contrast --n-boot 200 --boot-start 0 --boot-stop 20
lcsa e4 --cache C --out A --stage sweep --provo-dir data/provo --reference gpt2-large=R_DIR --tilted-from A
lcsa e4 --cache C --out A --stage boot --provo-dir data/provo --reference gpt2-large=R_DIR \
    --tilted-from A --n-boot 200 --boot-start 0 --boot-stop 20
lcsa merge --out A --legs e2,e3,e4
```

Shards land under `A/shards/` as JSON files named by their replicate range, and
`merge` refuses a set of shards that overlaps, has a gap, or does not start at
zero. The E2 ladder stage writes the nuisance vector to `e2_theta0.json` and
the E3 prepare stage writes `e3_prepared.npz`, so no shard refits anything the
stage already fitted.

### Reference caches

The pre-registration sweeps the context-limitation curve across five
zero-decay references and generates the recovery ladder under GPT-2-large.
`lcsa build --candidates B/candidates.json --targets B/targets.csv --model M`
builds a cache under another checkpoint on the primary build's frozen
candidate sets, so its rows align with the primary's and `--gen-cache` on `e2`
and `--reference NAME=DIR` on `e4` can use it. `--tilted-from A` adds the
N-TOPIC and N-ORDER tilts that `e3 --stage prepare` calibrated as two further
references at no GPU cost. The self-reference certification of the plain floor
is `lcsa e3 --cache GPT2_SMALL_CACHE --nulls none --readers N0 --no-human`.

## Kaggle

Two notebooks under `notebooks/` split along the same seam. Run `01_build_gpu.ipynb`
on a T4 x2 session to produce the cache, save it as a Kaggle dataset, then run
`02_estimate_cpu.ipynb` on a CPU session against that dataset. The build is the only
part that needs the accelerator, and separating them keeps the expensive step from
being repeated every time an estimation setting changes.

Measured against the paper's budget, the build is 61,435 distinct context forward
passes and costs about 5.35 GPU-hours at a 2.5 TFLOP/s floor, which is what two T4s
deliver with eager attention on Turing. The roughly 30,000 CPU fits that follow cost
about 90 core-hours, which is why `--n-rep` and `--n-boot` are exposed and why a
smoke run at 20 replicates is a reasonable first pass.

## Slurm

`slurm/` holds job scripts for the SCU cluster: the GPU build and the three
reference builds on `scu-gpu`, the E3 prepare stage on a GPU, and every
replicate loop as a `scu-cpu` array of shards, chained with `afterok` by
`slurm/pipeline.sh` and combined by `merge.sbatch`. At the default shard
counts the registered run finishes in about half a day of wall clock after the
builds. `slurm/README.md` explains the partition and QoS constraints the
scripts encode and the order to run them in.

## Repository layout

`src/lcsa/kernels.py` holds the retention kernel and its analytic derivative.
`src/lcsa/cache.py` builds the nested ablation cache, including the packed
block-diagonal forward pass that scores many depths in one batch and the fallback
that runs when a model wrapper ignores the 4-D attention mask.
`src/lcsa/likelihood.py` defines both estimators, their scores and their
information. `src/lcsa/projection.py` computes the nuisance span by thin QR in
whitened coordinates and decomposes a mismatch vector against it.
`src/lcsa/inference.py` has the cluster-robust score test, the profile regions, the
paired cluster bootstrap and the equivalence test. `src/lcsa/experiments/` holds one
module per leg, and `src/lcsa/gates.py` holds the pre-registered gates with their
fallbacks.

## Reproducibility

Every command takes `--seed` and every stochastic step draws from a seeded
generator, so a rerun with the same seed and the same cache reproduces the artifacts
byte for byte. Caches are saved as plain npz arrays with no pickled objects, so a
cache built on one machine loads on another regardless of the library versions
present.

## Licence

MIT. Provo and SUBTLEX-US carry their own terms and are not included.
