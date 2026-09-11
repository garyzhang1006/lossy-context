"""E2: the recovery ladder, interval coverage and the identification ceiling.

The ladder asks a question the nulls cannot: when decay is really there, at a
known depth, does the estimator find it, and does its interval mean what it
says?  A rung whose region runs off the top of the grid is not a failure of the
fit but a statement about the corpus, since a reader who forgets after 128 words
leaves almost no trace in contexts averaging 22 words long.  The largest rung
whose region is bounded above is what we call the identification ceiling.
"""

from __future__ import annotations

import logging

import numpy as np

from lcsa.corpusdata import Corpus
from lcsa.fitting import (ProfileRegion, default_grid, fit, local_grid,
                          profile_interval)
from lcsa.gates import g6_coverage
from lcsa.inference import score_test
from lcsa.kernels import POWER, d_half_from_delta, delta_from_d_half
from lcsa.likelihood import Model, information
from lcsa.readers import LADDER_DHALF, reader_ladder
from lcsa.experiments import Artifacts
from lcsa.experiments.shards import denull, read_shards, write_shard

log = logging.getLogger(__name__)

__all__ = ["coarse_grid", "delta_se", "fit_rung", "ladder_table", "coverage_replicates",
           "summarise_coverage", "coverage_at_rung", "identification_ceiling",
           "run_ladder", "run_coverage_shard", "assemble", "merge", "run"]


def coarse_grid(n: int = 9) -> np.ndarray:
    """A fixed 9-point profile grid, kept as the non-adaptive alternative.

    Coverage needs 200 replicates at two rungs under two estimators, so the grid
    is coarsened by design; the single reported region per rung keeps the full
    21 points.  A coarse grid can only widen a region, so coverage measured on
    it is not optimistic.
    """
    return default_grid(n=n)


def delta_se(corpus: Corpus, theta: np.ndarray, model: Model, kernel=POWER,
             design_effect: float = 1.0) -> float:
    """Cluster-corrected Wald standard error of ``delta`` at a fitted point.

    Used only to place the profile grid: the reported region is the profile one,
    and this number never appears in a result.  Multiplying by the square root of
    the design effect keeps the grid wide enough when responses within a passage
    are dependent.
    """
    I = information(corpus, theta, model, kernel)
    try:
        var = float(np.linalg.pinv(I)[0, 0])
    except np.linalg.LinAlgError:
        return float("nan")
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    d = float(design_effect) if np.isfinite(design_effect) and design_effect > 0 else 1.0
    return float(np.sqrt(var * d))


def fit_rung(
    gen_corpus: Corpus,
    fit_corpus_template: Corpus,
    d_half: float,
    theta_nuisance: np.ndarray,
    model: Model,
    kernel=POWER,
    seed: int = 0,
    grid=None,
    profile: bool = True,
    gen_model: Model | None = None,
    counts: list[np.ndarray] | None = None,
) -> dict:
    """Generate one rung's counts, fit it, and profile ``delta``.

    ``counts`` skips the generation, which is how the crossed panel of E6
    fits counts it drew with a tilt through the same code path.

    ``gen_corpus`` carries the generating cache and ``fit_corpus_template`` the
    fitting cache; in the paper these are different checkpoints, so recovery is
    measured under misspecification rather than in the self-reference case that
    always looks good.  ``gen_model`` is the estimator ``theta_nuisance`` was
    fitted under; the same synthetic reader is then fitted by every estimator,
    which is what makes the rows of one rung a paired comparison.
    """
    gen_model = model if gen_model is None else gen_model
    if counts is None:
        counts = reader_ladder(gen_corpus, d_half, theta_nuisance, gen_model, seed=seed,
                               kernel=kernel)
    corp = fit_corpus_template.with_counts(counts)
    f = fit(corp, model, kernel, n_starts=3, seed=seed)
    st = score_test(corp, model, kernel, n_starts=2, seed=seed)
    deff = st.design_effect
    scale = 1.0 / deff if np.isfinite(deff) and deff > 0 else 1.0
    row = {
        "d_half_true": float(d_half),
        "delta_true": float(delta_from_d_half(d_half, kernel=kernel)),
        "estimator": model.name,
        "delta_hat": float(f.delta),
        "d_half_hat": float(d_half_from_delta(f.delta, kernel=kernel)) if f.delta > 0 else float("inf"),
        "loglik": float(f.loglik),
        "converged": bool(f.success),
        "at_bound": bool(f.at_bound),
        "design_effect": float(deff),
        "p_score": float(st.p),
        "seed": int(seed),
    }
    if profile:
        if isinstance(grid, str) and grid == "local":
            g = local_grid(f.delta, se=delta_se(corp, f.theta, model, kernel, deff), n=9)
        else:
            g = grid
        reg = profile_interval(corp, model, kernel, grid=g, scale=scale,
                               n_starts=1, seed=seed, max_loglik=f.loglik, warm=f.theta)
        row.update({
            "delta_lo": float(reg.lo), "delta_hi": float(reg.hi),
            "unbounded_lo": bool(reg.unbounded_lo), "unbounded_hi": bool(reg.unbounded_hi),
            "d_half_lo": float(d_half_from_delta(reg.hi, kernel=kernel)) if reg.hi > 0 else float("inf"),
            "d_half_hi": float(d_half_from_delta(reg.lo, kernel=kernel)) if reg.lo > 0 else float("inf"),
            "covers_truth": bool(
                (reg.lo <= delta_from_d_half(d_half, kernel=kernel) <= reg.hi)
                or (reg.unbounded_hi and delta_from_d_half(d_half, kernel=kernel) >= reg.lo)
            ),
        })
    return row


def ladder_table(
    gen_corpus: Corpus,
    fit_corpus_template: Corpus,
    theta_nuisance: np.ndarray,
    models,
    rungs=LADDER_DHALF,
    kernel=POWER,
    seed: int = 0,
    gen_model: Model | None = None,
) -> list[dict]:
    """One profiled fit per rung per estimator, on the full 21-point grid."""
    gen_model = models[0] if gen_model is None else gen_model
    rows = []
    for model in models:
        for i, d in enumerate(rungs):
            try:
                rows.append(fit_rung(gen_corpus, fit_corpus_template, float(d),
                                     theta_nuisance, model, kernel, seed=seed + 17 * i,
                                     gen_model=gen_model))
            except Exception as exc:
                log.exception("rung d_half=%s under %s failed", d, model.name)
                rows.append({"d_half_true": float(d), "estimator": model.name,
                             "error": str(exc)})
    return rows


def coverage_replicates(
    gen_corpus: Corpus,
    fit_corpus_template: Corpus,
    d_half: float,
    theta_nuisance: np.ndarray,
    model: Model,
    reps=range(200),
    kernel=POWER,
    seed: int = 0,
    grid="local",
    progress=None,
    gen_model: Model | None = None,
) -> list[dict]:
    """One row per coverage replicate in ``reps``, seeded by the absolute index.

    Replicate ``b`` is generated and fitted from ``seed + b + 1`` whatever range
    it is computed in, which is what lets a Slurm array compute disjoint ranges
    and ``lcsa merge`` concatenate them into the numbers a single loop gives.
    """
    g = "local" if grid is None else grid
    rows = []
    reps = list(reps)
    for i, b in enumerate(reps):
        try:
            row = fit_rung(gen_corpus, fit_corpus_template, d_half, theta_nuisance,
                           model, kernel, seed=seed + b + 1, grid=g, gen_model=gen_model)
            row.update({"replicate": int(b), "failed": False})
        except Exception as exc:
            log.debug("coverage replicate %d at d_half=%s failed: %s", b, d_half, exc)
            row = {"d_half_true": float(d_half), "estimator": model.name,
                   "replicate": int(b), "failed": True, "error": str(exc)}
        rows.append(row)
        if progress is not None:
            progress(i + 1, len(reps))
    return rows


def summarise_coverage(rows: list[dict]) -> list[dict]:
    """Coverage per (estimator, rung) from per-replicate rows, in first-seen order.

    A replicate whose region is unbounded above still counts as covering when
    the truth lies above its lower limit, because that is what the region
    claims; counting it as a miss would flatter the estimator.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((str(r["estimator"]), float(r["d_half_true"])), []).append(r)
    out = []
    for (est, d), rs in groups.items():
        fails = sum(1 for r in rs if r.get("failed"))
        ok = [r for r in rs if not r.get("failed")]
        hit = sum(int(bool(r.get("covers_truth"))) for r in ok)
        unb = sum(int(bool(r.get("unbounded_hi"))) for r in ok)
        used = len(ok)
        out.append({
            "d_half_true": d,
            "estimator": est,
            "n_replicates": int(len(rs)),
            "n_usable": int(used),
            "n_failed": int(fails),
            "coverage": float(hit / used) if used else float("nan"),
            "frac_unbounded_above": float(unb / used) if used else float("nan"),
        })
    return out


def coverage_at_rung(
    gen_corpus: Corpus,
    fit_corpus_template: Corpus,
    d_half: float,
    theta_nuisance: np.ndarray,
    model: Model,
    n_rep: int = 200,
    kernel=POWER,
    seed: int = 0,
    grid="local",
    progress=None,
) -> dict:
    """Empirical coverage of the nominal 95 percent region at one rung."""
    rows = coverage_replicates(gen_corpus, fit_corpus_template, d_half, theta_nuisance,
                               model, range(n_rep), kernel, seed, grid, progress)
    return summarise_coverage(rows)[0]


#: Rungs at which coverage replicates are run.  4 and 8 feed gate G6; the
#: rest are the ceiling rungs, replicated so that "bounded above" is a rate.
COVERAGE_RUNGS = (4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0)
#: A rung counts as identified when at least this share of replicates bound it.
CEILING_SHARE = 0.5


def identification_ceiling(rows: list[dict], estimator: str | None = None,
                           n_boot: int = 200, seed: int = 0, share: float = CEILING_SHARE) -> dict:
    """The ceiling as a rate: the largest rung bounded above in at least half the replicates.

    ``rows`` are per-replicate coverage rows.  A single ladder cannot say
    whether the region at 16 is bounded above by luck, so the ceiling is read
    from the share of replicates whose region is bounded above at each rung,
    and its interval comes from a bootstrap that resamples replicates within
    each rung and re-reads the ceiling.  Ladder rows without ``replicate`` are
    accepted too, in which case the share is one or zero per rung and the
    interval collapses, which is the honest answer for one draw.
    """
    sel = [r for r in rows if "d_half_true" in r and "unbounded_hi" in r
           and not r.get("failed") and np.isfinite(r["d_half_true"])
           and (estimator is None or r.get("estimator") == estimator)]
    by_rung: dict[float, list[bool]] = {}
    for r in sel:
        by_rung.setdefault(float(r["d_half_true"]), []).append(not bool(r["unbounded_hi"]))
    rungs = sorted(by_rung)
    bounded_share = {d: float(np.mean(by_rung[d])) for d in rungs}

    def _ceiling(shares: dict[float, float]) -> float:
        ok = [d for d in rungs if shares[d] >= share]
        return float(max(ok)) if ok else float("nan")

    ceiling = _ceiling(bounded_share)
    draws = []
    for b in range(n_boot):
        rng = np.random.default_rng([int(seed), 31, int(b)])
        boot = {}
        for d in rungs:
            v = np.asarray(by_rung[d], dtype=np.float64)
            boot[d] = float(np.mean(v[rng.integers(0, v.size, size=v.size)]))
        draws.append(_ceiling(boot))
    draws = np.asarray(draws)
    fin = draws[np.isfinite(draws)]
    lo, hi = (np.percentile(fin, [2.5, 97.5]) if fin.size > 3 else (float("nan"), float("nan")))
    first_below = next((d for d in rungs if bounded_share[d] < share), None)
    return {
        "estimator": estimator,
        "ceiling_d_half": ceiling,
        "ceiling_lo": float(lo), "ceiling_hi": float(hi),
        "share_bounded_by_rung": {str(d): bounded_share[d] for d in rungs},
        "n_replicates_by_rung": {str(d): len(by_rung[d]) for d in rungs},
        "first_rung_below_share": float(first_below) if first_below is not None else None,
        "share_threshold": float(share),
        "n_boot": int(n_boot),
        "in_registered_window_12_to_30": bool(np.isfinite(ceiling) and 12.0 <= ceiling <= 30.0),
    }


def run_ladder(
    gen_corpus: Corpus,
    fit_corpus_template: Corpus,
    theta_nuisance: np.ndarray,
    models,
    out_dir,
    kernel=POWER,
    seed: int = 0,
) -> list[dict]:
    """Stage one of E2: the eleven-rung ladder on the full grid, written to disk."""
    art = Artifacts(out_dir, "e2")
    rows = ladder_table(gen_corpus, fit_corpus_template, theta_nuisance, models,
                        kernel=kernel, seed=seed, gen_model=models[0])
    art.table("e2_ladder", rows)
    art.save("e2_ladder", rows)
    return rows


def run_coverage_shard(
    gen_corpus: Corpus,
    fit_corpus_template: Corpus,
    theta_nuisance: np.ndarray,
    models,
    out_dir,
    reps: range,
    kernel=POWER,
    coverage_rungs=COVERAGE_RUNGS,
    seed: int = 0,
) -> list[dict]:
    """Stage two of E2: coverage replicates ``reps`` at every rung and estimator."""
    rows = []
    for model in models:
        for d in coverage_rungs:
            rows += coverage_replicates(gen_corpus, fit_corpus_template, float(d),
                                        theta_nuisance, model, reps, kernel=kernel,
                                        seed=seed, gen_model=models[0])
    write_shard(out_dir, "e2_coverage", reps, rows)
    return rows


def assemble(ladder_rows: list[dict], coverage_rows: list[dict], models, out_dir,
             coverage_rungs=(4.0, 8.0)) -> dict:
    """Stage three of E2: the coverage table, gate G6 and the ceiling as a rate.

    G6 is gated on the two shortest rungs only, whatever ``coverage_rungs``
    holds, because the longer rungs exist to place the ceiling and a ceiling
    is a finding, not a gate.
    """
    art = Artifacts(out_dir, "e2")
    cov_rows = summarise_coverage(coverage_rows)
    art.table("e2_coverage", cov_rows)
    primary = models[0].name if hasattr(models[0], "name") else str(models[0])
    names = [m.name if hasattr(m, "name") else str(m) for m in models]
    cov_map = {r["d_half_true"]: r["coverage"] for r in cov_rows if r["estimator"] == primary}
    gate_rungs = tuple(d for d in (4.0, 8.0) if d in cov_map) or tuple(coverage_rungs[:2])
    res = {
        "ladder": ladder_rows,
        "coverage": cov_rows,
        "g6": g6_coverage(cov_map, rungs=gate_rungs),
        "ceiling": [identification_ceiling(coverage_rows, m) for m in names],
        "ceiling_single_ladder": [
            {"estimator": m,
             "ceiling_d_half": identification_ceiling(ladder_rows, m, n_boot=0)["ceiling_d_half"]}
            for m in names],
    }
    art.save("e2_summary", res)
    return res


def merge(out_dir, models, coverage_rungs=(4.0, 8.0), n_rep: int | None = None) -> dict:
    """Combine ``e2_ladder.json`` and the coverage shards into the registered tables."""
    import json
    from pathlib import Path

    p = Path(out_dir) / "e2_ladder.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; run `lcsa e2 --stage ladder` first")
    ladder = denull(json.loads(p.read_text()))
    return assemble(ladder, read_shards(out_dir, "e2_coverage", n_rep), models, out_dir,
                    coverage_rungs)


def run(
    gen_corpus: Corpus,
    fit_corpus_template: Corpus,
    theta_nuisance: np.ndarray,
    models,
    out_dir,
    kernel=POWER,
    n_rep: int = 200,
    coverage_rungs=COVERAGE_RUNGS,
    seed: int = 0,
) -> dict:
    """Full E2 leg in one process: the ladder, the coverage rungs, and the ceiling."""
    rows = run_ladder(gen_corpus, fit_corpus_template, theta_nuisance, models, out_dir,
                      kernel=kernel, seed=seed)
    cov = run_coverage_shard(gen_corpus, fit_corpus_template, theta_nuisance, models,
                             out_dir, range(n_rep), kernel=kernel,
                             coverage_rungs=coverage_rungs, seed=seed)
    return assemble(rows, cov, models, out_dir, coverage_rungs)
