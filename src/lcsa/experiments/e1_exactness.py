"""E1: exactness, sensitivity, the residual fractions and the bridging experiment.

Nothing in this file needs the human counts, and most of it needs no data at
all.  The exactness checks are arithmetic identities that either hold to machine
precision or reveal a bug; the sensitivity profile is an analytic ceiling on how
deep any estimator could recover retention from this corpus; the residual
fractions are the empirical content of the absorption criterion.  The bridging
experiment is the only part that costs GPU time, because it needs every subset
of a short context rather than every suffix.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np

from lcsa.corpusdata import Corpus, PROB_FLOOR
from lcsa.fitting import fit
from lcsa.gates import g2_sensitivity
from lcsa.kernels import (POWER, delta_from_d_half, marginalise, marginalise_abel,
                          marginalise_and_grad, retention, truncation_weights)
from lcsa.likelihood import Model, evaluate_target, loglik_and_grad
from lcsa.projection import corpus_residual_fraction, global_residual
from lcsa.experiments import Artifacts

log = logging.getLogger(__name__)

__all__ = ["exactness_report", "likelihood_gradient_check", "sensitivity_by_distance", "subset_masks",
           "independent_weights", "independent_marginal", "graded_marginal_from_subset",
           "bridging_report", "residual_table", "run"]


def exactness_report(kernel=POWER, deltas=(0.0, 0.1, 0.316, 0.631, 1.0, 2.0),
                     K: int = 32, seed: int = 0) -> dict:
    """The identities: weights sum to one, reproduce ``r``, and the score is analytic.

    The tail sum of the truncation atoms from ``m`` upward is exactly ``r(m)``,
    which is the statement that graded truncation has the per-position marginal
    it advertises.  Checking it at every grid point costs microseconds and would
    have caught a sign error in the atom construction that no downstream test
    could localise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in deltas:
        w = truncation_weights(K, float(d), kernel)
        r = retention(np.arange(K + 2), float(d), kernel)
        tail = np.array([w[m:].sum() for m in range(K + 1)])
        rows.append({
            "delta": float(d),
            "weight_sum_error": float(abs(w.sum() - 1.0)),
            "min_weight": float(w.min()),
            "max_tail_vs_retention_error": float(np.max(np.abs(tail - r[: K + 1]))),
            "full_retention_at_zero": float(np.max(np.abs(r[: K + 1] - 1.0))) if d == 0 else None,
        })

    V = 40
    P = rng.random((K + 1, V)) + 0.05
    P /= P.sum(axis=1, keepdims=True)
    D = np.diff(P, axis=0)
    atom_abel, grad_err = [], []
    for d in deltas:
        a = marginalise(P, float(d), kernel)
        b = marginalise_abel(P, float(d), kernel)
        atom_abel.append(float(np.max(np.abs(a - b))))
        _, g = marginalise_and_grad(P, float(d), kernel, D)
        h = 1e-6
        fd = (marginalise(P, float(d) + h, kernel) - marginalise(P, max(float(d) - h, 0.0), kernel))
        fd = fd / ((float(d) + h) - max(float(d) - h, 0.0))
        grad_err.append(float(np.max(np.abs(g - fd))))
    return {
        "per_delta": rows,
        "max_atom_vs_abel_error": float(max(atom_abel)),
        "max_score_vs_finite_difference": float(max(grad_err)),
        "K": int(K),
        "kernel": kernel.name,
        "passed": (max(atom_abel) < 1e-12
                   and max(grad_err) < 1e-5
                   and max(r["weight_sum_error"] for r in rows) < 1e-12),
    }


def likelihood_gradient_check(corpus: Corpus, model: Model, kernel=POWER,
                              seed: int = 0, h: float = 1e-6) -> dict:
    """Central-difference check of every coordinate of the analytic score."""
    rng = np.random.default_rng(seed)
    theta = model.start(corpus.M).copy()
    theta[0] = 0.3
    theta = theta + 0.01 * rng.standard_normal(theta.size)
    theta[0] = abs(theta[0])
    theta[1] = min(max(theta[1], 1e-3), 0.5)
    theta[2] = abs(theta[2]) + 0.5
    _, g = loglik_and_grad(corpus, theta, model, kernel)
    fd = np.empty_like(g)
    for i in range(theta.size):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += h
        tm[i] -= h
        Lp, _ = loglik_and_grad(corpus, tp, model, kernel)
        Lm, _ = loglik_and_grad(corpus, tm, model, kernel)
        fd[i] = (Lp - Lm) / (2 * h)
    denom = np.maximum(np.abs(g), 1.0)
    rel = float(np.max(np.abs(g - fd) / denom))
    return {"model": model.name, "max_relative_error": rel, "passed": rel < 1e-4,
            "analytic": g.tolist(), "finite_difference": fd.tolist()}


def sensitivity_by_distance(corpus: Corpus, max_j: int = 32) -> list[dict]:
    """Total-variation norm of each incremental displacement against distance."""
    rows = []
    for j in range(1, max_j + 1):
        tv = [0.5 * float(np.abs(t.D[j - 1]).sum()) for t in corpus if t.K >= j]
        if not tv:
            continue
        a = np.asarray(tv)
        rows.append({
            "j": j,
            "n_contexts": int(a.size),
            "mean_tv": float(a.mean()),
            "median_tv": float(np.median(a)),
            "frac_above_0.02": float((a >= 0.02).mean()),
            "frac_above_0.005": float((a >= 0.005).mean()),
        })
    return rows


def subset_masks(K: int) -> list[int]:
    """All ``2^K`` retention masks as bitmasks, bit ``j-1`` meaning distance ``j`` kept."""
    if K > 12:
        raise ValueError(f"enumerating 2^{K} masks is not the intended use; K <= 12")
    return list(range(1 << K))


def independent_weights(K: int, delta: float, kernel=POWER) -> np.ndarray:
    """``P(mask)`` under independent deletion, in the bitmask order of :func:`subset_masks`."""
    r = retention(np.arange(1, K + 1), float(delta), kernel)
    w = np.empty(1 << K, dtype=np.float64)
    for m in range(1 << K):
        p = 1.0
        for j in range(K):
            p *= r[j] if (m >> j) & 1 else (1.0 - r[j])
        w[m] = p
    return w


def independent_marginal(sub_P: np.ndarray, K: int, delta: float, kernel=POWER) -> np.ndarray:
    """Marginal predictive under independent deletion, exact over all subsets."""
    w = independent_weights(K, delta, kernel)
    return w @ sub_P


def graded_marginal_from_subset(sub_P: np.ndarray, K: int, delta: float,
                                kernel=POWER) -> np.ndarray:
    """Graded-truncation marginal read off the same subset cache.

    Graded truncation puts all its mass on the ``K + 1`` contiguous suffix masks,
    so this must agree with :func:`lcsa.kernels.marginalise` on the nested cache;
    computing it from the subset cache instead is what makes the comparison
    between kernels an apples-to-apples one.
    """
    w = truncation_weights(K, float(delta), kernel)
    idx = [((1 << k) - 1) for k in range(K + 1)]  # keep the k nearest positions
    return w @ sub_P[idx]


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), PROB_FLOOR, None)
    q = np.clip(np.asarray(q, dtype=np.float64), PROB_FLOOR, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def bridging_report(
    corpus: Corpus,
    sub_caches: dict[int, np.ndarray],
    model: Model,
    d_half_true: float = 8.0,
    kernel=POWER,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Prediction 10: how far apart are the two kernels, in nats and in fitted delta.

    ``sub_caches`` maps a corpus target index to its ``(2^K, V)`` subset cache.
    The two kernels put their mass on disjoint families of conditioning sets, so
    a large divergence between the predictive distributions is expected; the
    number that decides the scope of the paper's claims is the second one, the
    gap between the delta fitted under graded truncation and the delta that
    actually generated the responses under independent deletion.
    """
    delta_true = delta_from_d_half(float(d_half_true))
    idx = sorted(sub_caches)
    if not idx:
        raise ValueError("bridging needs at least one target with a subset cache")

    rng = np.random.default_rng(seed)
    kls, tvs, counts, clusters = [], [], [], []
    for t in idx:
        tgt = corpus.target(t)
        sub = np.asarray(sub_caches[t], dtype=np.float64)
        K = int(np.log2(sub.shape[0]))
        if sub.shape != (1 << K, tgt.V):
            raise ValueError(
                f"target {t}: subset cache has shape {sub.shape}, expected {(1 << K, tgt.V)}"
            )
        p_ind = independent_marginal(sub, K, delta_true, kernel)
        p_grd = graded_marginal_from_subset(sub, K, delta_true, kernel)
        kls.append(_kl(p_ind, p_grd))
        tvs.append(0.5 * float(np.abs(p_ind - p_grd).sum()))
        N = int(round(tgt.N))
        counts.append(rng.multinomial(N, p_ind / p_ind.sum()).astype(np.float64)
                      if N > 0 else np.zeros(tgt.V))
        clusters.append(int(tgt.cluster))

    small = Corpus(
        [corpus.target(t).P for t in idx],
        counts,
        [corpus.target(t).u for t in idx],
        [corpus.target(t).f for t in idx],
        [corpus.target(t).g for t in idx],
        clusters,
        word_ids=[corpus.target(t).word_ids for t in idx],
        feature_names=corpus.feature_names,
        target_slots=[corpus.target(t).target_slot for t in idx],
    )
    f = fit(small, model, kernel, n_starts=3, seed=seed)

    reps = []
    cl = np.unique(small.cluster_index)
    for b in range(n_boot):
        pick = np.random.default_rng(seed + b + 1).choice(cl, size=cl.size, replace=True)
        try:
            fb = fit(small.subset_clusters(pick), model, kernel, n_starts=1, seed=seed)
        except Exception as exc:  # a boundary fit is data, not an error to hide
            log.debug("bridging bootstrap replicate %d failed: %s", b, exc)
            reps.append(np.nan)
            continue
        reps.append(np.log(fb.delta) - np.log(delta_true) if fb.delta > 0 else -np.inf)
    r = np.asarray(reps, dtype=np.float64)
    fin = r[np.isfinite(r)]
    return {
        "n_targets": len(idx),
        "n_clusters": int(small.n_clusters),
        "d_half_true": float(d_half_true),
        "delta_true": float(delta_true),
        "mean_kl_indep_given_graded": float(np.mean(kls)),
        "median_kl_indep_given_graded": float(np.median(kls)),
        "mean_tv_between_kernels": float(np.mean(tvs)),
        "delta_hat_graded": float(f.delta),
        "log_delta_gap": float(np.log(f.delta) - np.log(delta_true)) if f.delta > 0 else None,
        "log_delta_gap_ci": (
            [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))]
            if fin.size >= 20 else None
        ),
        "n_boot_usable": int(fin.size),
        "n_boot": int(n_boot),
        "agrees_within_0.25_log_units": bool(
            f.delta > 0 and abs(np.log(f.delta) - np.log(delta_true)) <= 0.25
        ),
    }


def residual_table(readers: dict[str, Corpus], models, kernel=POWER,
                   alphas=(0.1, 0.5, 1.0)) -> list[dict]:
    """``||h_perp||_N / ||h||_N`` per reader per estimator, at the constrained null.

    The global fraction projects onto the tangent space of one shared nuisance
    vector and is the quantity of Proposition 2; the per-target fraction lets
    every context choose its own coefficients and is a lower bound.  Neither
    depends on a fit of ``delta``, which is the point: both can be computed
    and reported before the human counts are unfrozen.
    """
    from lcsa.fitting import fit_constrained

    rows = []
    for name, corp in readers.items():
        for model in models:
            try:
                null = fit_constrained(corp, model, kernel, n_starts=2)
            except Exception as exc:
                rows.append({"reader": name, "estimator": model.name, "error": str(exc)})
                continue
            for a in alphas:
                rep = corpus_residual_fraction(corp, null.theta, model, kernel, alpha=a)
                g = global_residual(corp, null.theta, model, kernel, alpha=a)
                rows.append({
                    "reader": name,
                    "estimator": model.name,
                    "jeffreys_alpha": float(a),
                    "residual_fraction_global": g.fraction,
                    "residual_fraction_per_target": rep.fraction,
                    "residual_fraction_unweighted": rep.fraction_unweighted,
                    "alignment": g.alignment,
                    "span_dim_mean": rep.span_dim_mean,
                    "n_targets": rep.n_targets_used,
                    "nuisance_at_bound": bool(null.nuisance_at_bound),
                })
    return rows


def run(corpus: Corpus, models, out_dir, kernel=POWER, sub_caches=None,
        d_half_true: float = 8.0, seed: int = 0, prefix_probe=None) -> dict:
    """Full E1 leg: exactness, gradients, sensitivity, G2, residuals, bridging.

    ``prefix_probe`` is the summary block of ``prefix_probe.json``; the probe
    itself needs the reference on a GPU, so it runs as its own command and the
    leg only folds its summary into ``e1_summary`` for prediction 12 to score.
    """
    art = Artifacts(out_dir, "e1")
    res = {"exactness": exactness_report(kernel=kernel)}
    res["gradients"] = [likelihood_gradient_check(corpus, m, kernel, seed=seed) for m in models]
    sens = sensitivity_by_distance(corpus)
    art.table("e1_sensitivity", sens)
    res["sensitivity"] = sens
    res["g2"] = g2_sensitivity(corpus)
    rows = residual_table({"human": corpus}, models, kernel)
    art.table("e1_residual_fraction", rows)
    res["residual_fraction"] = rows
    if sub_caches:
        res["bridging"] = bridging_report(corpus, sub_caches, models[0], d_half_true,
                                          kernel, seed=seed)
    if prefix_probe:
        res["prefix_probe"] = dict(prefix_probe)
    art.save("e1_summary", res)
    return res
