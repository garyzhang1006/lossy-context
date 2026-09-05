"""Model-free descriptives and the hard-truncation sweep the field already runs.

Two things live here.  The first is a two-number summary of the human data that
needs no estimator at all: how the entropy of the cloze responses and the
agreement of their modal answer move with the number of preceding words a reader
actually had.  If neither moves, no likelihood is going to find context decay,
and saying so with a slope is cheaper and more honest than saying it with a
fitted parameter.

The second is the context-limitation sweep of Kuribayashi et al. (2022) in the
form this paper audits: hard truncation at a fixed window, selected by AIC and
by held-out log-likelihood with passages as folds.  Running it under readers
that have no decay whatsoever is what turns prediction 8 into a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lcsa.corpusdata import Corpus, PROB_FLOOR
from lcsa.fitting import fit
from lcsa.kernels import POWER
from lcsa.likelihood import Model, loglik

__all__ = ["SlopeStat", "context_slopes", "auc", "hard_window_corpus",
           "WindowFit", "hard_window_sweep", "sweep_disagreement"]


@dataclass
class SlopeStat:
    slope: float
    se: float
    t: float
    n: int
    n_clusters: int
    intercept: float

    @property
    def significant(self) -> bool:
        return bool(np.isfinite(self.t) and abs(self.t) > 2.0)


def _cluster_ols(y: np.ndarray, x: np.ndarray, cluster: np.ndarray) -> SlopeStat:
    """Simple regression of ``y`` on ``x`` with a passage-clustered slope SE."""
    ok = np.isfinite(y) & np.isfinite(x)
    y, x, cluster = y[ok], x[ok], cluster[ok]
    n = y.size
    if n < 3 or np.ptp(x) == 0:
        return SlopeStat(float("nan"), float("nan"), float("nan"), int(n), 0, float("nan"))
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    uniq = np.unique(cluster)
    meat = np.zeros((2, 2))
    for c in uniq:
        m = cluster == c
        sc = X[m].T @ resid[m]
        meat += np.outer(sc, sc)
    C = uniq.size
    if C > 1:
        meat *= C / (C - 1.0)
    V = XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(max(V[1, 1], 0.0)))
    return SlopeStat(float(beta[1]), se, float(beta[1] / se) if se > 0 else float("nan"),
                     int(n), int(C), float(beta[0]))


def context_slopes(corpus: Corpus) -> dict:
    """Entropy and top-1 agreement of the human responses against available context.

    Both quantities are computed from the response counts alone, so nothing here
    depends on the reference model, the candidate set beyond its support, or any
    fitted parameter.  A flat entropy slope with a large fitted decay is a
    contradiction worth printing.
    """
    H, top1, K, cl, N = [], [], [], [], []
    for tgt in corpus:
        tot = tgt.n.sum()
        if tot <= 0:
            continue
        p = tgt.n / tot
        nz = p[p > 0]
        H.append(float(-(nz * np.log(nz)).sum()))
        top1.append(float(p.max()))
        K.append(float(tgt.K))
        cl.append(int(tgt.cluster))
        N.append(float(tot))
    H = np.asarray(H); top1 = np.asarray(top1)
    K = np.asarray(K); cl = np.asarray(cl); N = np.asarray(N)
    return {
        "entropy": _cluster_ols(H, K, cl),
        "top1": _cluster_ols(top1, K, cl),
        "mean_entropy": float(H.mean()) if H.size else float("nan"),
        "mean_top1": float(top1.mean()) if top1.size else float("nan"),
        "mean_K": float(K.mean()) if K.size else float("nan"),
        "mean_responses": float(N.mean()) if N.size else float("nan"),
        "n_targets": int(H.size),
    }


def auc(a, b) -> float:
    """Rank-based separation of two score samples, ties counted as half.

    Returns the probability that a draw from ``a`` exceeds a draw from ``b``.
    0.5 means the two readers are indistinguishable on that score, which is the
    interesting outcome when ``a`` is human and ``b`` is a zero-decay null.
    """
    a = np.asarray([x for x in np.asarray(a, dtype=float).ravel() if np.isfinite(x)])
    b = np.asarray([x for x in np.asarray(b, dtype=float).ravel() if np.isfinite(x)])
    if a.size == 0 or b.size == 0:
        return float("nan")
    allv = np.concatenate([a, b])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(allv.size, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1, dtype=np.float64)
    # Average ranks within ties so that a constant score gives exactly 0.5.
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    ra = ranks[: a.size].sum()
    return float((ra - a.size * (a.size + 1) / 2.0) / (a.size * b.size))


def hard_window_corpus(corpus: Corpus, window: int) -> Corpus:
    """A copy whose cache is the single row for hard truncation at ``window`` words.

    Collapsing the cache to one depth makes ``delta`` inert by construction, so
    the sweep fits exactly the model the context-limitation literature fits: a
    fixed window with the same production nuisances and no graded retention.
    """
    if window < 0:
        raise ValueError(f"window must be non-negative, got {window}")
    P_list, n_list, u_list, f_list, g_list, cl, wid, slots = [], [], [], [], [], [], [], []
    for tgt in corpus:
        j = min(int(window), tgt.K)
        P_list.append(tgt.P[j][None, :].copy())
        n_list.append(tgt.n.copy())
        u_list.append(tgt.u.copy())
        f_list.append(tgt.f.copy())
        g_list.append(tgt.g.copy())
        cl.append(int(tgt.cluster))
        wid.append(tgt.word_ids.copy())
        slots.append(int(tgt.target_slot))
    return Corpus(P_list, n_list, u_list, f_list, g_list, cl,
                  word_ids=wid, feature_names=corpus.feature_names, target_slots=slots)


@dataclass
class WindowFit:
    window: int
    loglik: float
    n_params: int
    aic: float
    heldout_per_response: float
    converged: bool


def hard_window_sweep(
    corpus: Corpus,
    model: Model,
    windows=(0, 1, 2, 4, 8, 16, 32),
    n_folds: int = 5,
    n_starts: int = 2,
    seed: int = 0,
) -> list[WindowFit]:
    """Fit each fixed window, scoring by AIC and by held-out log-likelihood.

    Folds partition passages, never targets, so a window that wins only because
    it memorised a passage's vocabulary cannot win here.  ``n_folds`` at 55 is
    genuine leave-one-passage-out and costs 55 fits per window; the default of 5
    is the same estimator at a tenth of the compute.
    """
    rng = np.random.default_rng(seed)
    uniq = np.arange(corpus.n_clusters)
    perm = rng.permutation(uniq)
    folds = np.array_split(perm, min(max(n_folds, 2), uniq.size))
    P = model.dim(corpus.M)
    out: list[WindowFit] = []
    for w in windows:
        cw = hard_window_corpus(corpus, int(w))
        f = fit(cw, model, POWER, fixed={0: 0.0}, n_starts=n_starts, seed=seed)
        ho, resp = 0.0, 0.0
        for fold in folds:
            te = np.asarray(fold, dtype=int)
            tr = np.setdiff1d(uniq, te)
            if tr.size == 0 or te.size == 0:
                continue
            ftr = fit(cw.subset_clusters(tr), model, POWER, fixed={0: 0.0},
                      n_starts=1, seed=seed)
            cte = cw.subset_clusters(te)
            ho += float(loglik(cte, ftr.theta, model, POWER))
            resp += float(cte.total_responses)
        out.append(WindowFit(
            window=int(w),
            loglik=float(f.loglik),
            n_params=int(P - 1),
            aic=float(2 * (P - 1) - 2 * f.loglik),
            heldout_per_response=float(ho / resp) if resp > 0 else float("nan"),
            converged=bool(f.success),
        ))
    return out


def sweep_disagreement(sweeps: dict[str, list[WindowFit]]) -> dict:
    """Prediction 8 as a number: do zero-decay references pick the same window?

    ``sweeps`` maps a reference name to its window sweep.  The prediction is
    scored by the ratio of the largest to the smallest selected window across
    references, with the grid floor treated as a separate outcome because every
    reference selecting zero context is the other way the sweep can fail to be a
    measurement.
    """
    best_aic, best_ho = {}, {}
    for name, rows in sweeps.items():
        if not rows:
            continue
        best_aic[name] = min(rows, key=lambda r: r.aic).window
        finite = [r for r in rows if np.isfinite(r.heldout_per_response)]
        if finite:
            best_ho[name] = max(finite, key=lambda r: r.heldout_per_response).window
    vals = [v for v in best_aic.values()]
    pos = [v for v in vals if v > 0]
    ratio = (max(pos) / min(pos)) if len(pos) >= 2 else float("nan")
    return {
        "selected_by_aic": best_aic,
        "selected_by_heldout": best_ho,
        "max_min_ratio": float(ratio),
        "all_at_grid_floor": bool(vals) and all(v == 0 for v in vals),
        "disagrees_by_more_than_2x": bool(np.isfinite(ratio) and ratio > 2.0),
    }
