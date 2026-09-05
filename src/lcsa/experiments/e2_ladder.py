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

log = logging.getLogger(__name__)

__all__ = ["coarse_grid", "delta_se", "fit_rung", "ladder_table", "coverage_at_rung",
           "identification_ceiling", "run"]


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
) -> dict:
    """Generate one rung's counts, fit it, and profile ``delta``.

    ``gen_corpus`` carries the generating cache and ``fit_corpus_template`` the
    fitting cache; in the paper these are different checkpoints, so recovery is
    measured under misspecification rather than in the self-reference case that
    always looks good.
    """
    counts = reader_ladder(gen_corpus, d_half, theta_nuisance, model, seed=seed, kernel=kernel)
    corp = fit_corpus_template.with_counts(counts)
    f = fit(corp, model, kernel, n_starts=3, seed=seed)
    st = score_test(corp, model, kernel, n_starts=2, seed=seed)
    deff = st.design_effect
    scale = 1.0 / deff if np.isfinite(deff) and deff > 0 else 1.0
    row = {
        "d_half_true": float(d_half),
        "delta_true": float(delta_from_d_half(d_half)),
        "estimator": model.name,
        "delta_hat": float(f.delta),
        "d_half_hat": float(d_half_from_delta(f.delta)) if f.delta > 0 else float("inf"),
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
            "d_half_lo": float(d_half_from_delta(reg.hi)) if reg.hi > 0 else float("inf"),
            "d_half_hi": float(d_half_from_delta(reg.lo)) if reg.lo > 0 else float("inf"),
            "covers_truth": bool(
                (reg.lo <= delta_from_d_half(d_half) <= reg.hi)
                or (reg.unbounded_hi and delta_from_d_half(d_half) >= reg.lo)
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
) -> list[dict]:
    """One profiled fit per rung per estimator, on the full 21-point grid."""
    rows = []
    for model in models:
        for i, d in enumerate(rungs):
            try:
                rows.append(fit_rung(gen_corpus, fit_corpus_template, float(d),
                                     theta_nuisance, model, kernel, seed=seed + 17 * i))
            except Exception as exc:
                log.exception("rung d_half=%s under %s failed", d, model.name)
                rows.append({"d_half_true": float(d), "estimator": model.name,
                             "error": str(exc)})
    return rows


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
    """Empirical coverage of the nominal 95 percent region at one rung.

    A replicate whose region is unbounded above still counts as covering when
    the truth lies above its lower limit, because that is what the region
    claims; counting it as a miss would flatter the estimator.
    """
    g = "local" if grid is None else grid
    hit, unb, fails = 0, 0, 0
    for b in range(n_rep):
        try:
            row = fit_rung(gen_corpus, fit_corpus_template, d_half, theta_nuisance,
                           model, kernel, seed=seed + b + 1, grid=g)
        except Exception as exc:
            log.debug("coverage replicate %d at d_half=%s failed: %s", b, d_half, exc)
            fails += 1
            continue
        hit += int(bool(row.get("covers_truth")))
        unb += int(bool(row.get("unbounded_hi")))
        if progress is not None:
            progress(b + 1, n_rep)
    used = n_rep - fails
    return {
        "d_half_true": float(d_half),
        "estimator": model.name,
        "n_replicates": int(n_rep),
        "n_usable": int(used),
        "n_failed": int(fails),
        "coverage": float(hit / used) if used else float("nan"),
        "frac_unbounded_above": float(unb / used) if used else float("nan"),
    }


def identification_ceiling(rows: list[dict], estimator: str | None = None) -> dict:
    """The largest true ``d_half`` whose reported region is bounded above."""
    sel = [r for r in rows if "d_half_true" in r and "unbounded_hi" in r
           and (estimator is None or r.get("estimator") == estimator)]
    bounded = [r["d_half_true"] for r in sel
               if not r["unbounded_hi"] and np.isfinite(r["d_half_true"])]
    unbounded = [r["d_half_true"] for r in sel if r["unbounded_hi"]]
    return {
        "estimator": estimator,
        "ceiling_d_half": float(max(bounded)) if bounded else float("nan"),
        "first_unbounded_d_half": float(min(unbounded)) if unbounded else None,
        "bounded_rungs": sorted(float(x) for x in bounded),
        "in_registered_window_12_to_30": bool(
            bounded and 12.0 <= max(bounded) <= 30.0
        ),
    }


def run(
    gen_corpus: Corpus,
    fit_corpus_template: Corpus,
    theta_nuisance: np.ndarray,
    models,
    out_dir,
    kernel=POWER,
    n_rep: int = 200,
    coverage_rungs=(4.0, 8.0),
    seed: int = 0,
) -> dict:
    """Full E2 leg: the ladder, the two coverage rungs, and the ceiling."""
    art = Artifacts(out_dir, "e2")
    rows = ladder_table(gen_corpus, fit_corpus_template, theta_nuisance, models,
                        kernel=kernel, seed=seed)
    art.table("e2_ladder", rows)

    cov_rows = []
    for model in models:
        for d in coverage_rungs:
            cov_rows.append(coverage_at_rung(gen_corpus, fit_corpus_template, float(d),
                                             theta_nuisance, model, n_rep=n_rep,
                                             kernel=kernel, seed=seed))
    art.table("e2_coverage", cov_rows)

    primary = models[0].name
    cov_map = {r["d_half_true"]: r["coverage"] for r in cov_rows if r["estimator"] == primary}
    res = {
        "ladder": rows,
        "coverage": cov_rows,
        "g6": g6_coverage(cov_map, rungs=tuple(coverage_rungs)),
        "ceiling": [identification_ceiling(rows, m.name) for m in models],
    }
    art.save("e2_summary", res)
    return res
