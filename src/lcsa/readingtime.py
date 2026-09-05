"""Held-out reading-time gain from a fitted kernel, the external check of prediction 7.

Registered prediction 7 says a null's *spuriously* fitted kernel recovers at
least 60 percent of the human kernel's held-out reading-time gain.  If that
holds, reading-time fit is not diagnostic of a real memory parameter, because a
kernel fitted to a reader with no decay at all buys nearly as much predictive
gain.  Its falsification would make reading-time fit a genuine external check,
which the paper says it would be glad to report.

The gain is measured as the increase in held-out log-likelihood, per word, from
adding lossy-context surprisal at the fitted ``delta`` to a baseline that already
contains full-context surprisal and the usual controls.  Folds are over
*passages*, never over words, because words in a passage are not exchangeable
and a word-level fold would leak the passage's own random effect into training.

Random effects come from ``statsmodels`` MixedLM when it is available and
converges; otherwise the model degrades to OLS with cluster-robust standard
errors, and the returned record says which was used.  Silently substituting OLS
for a mixed model would understate uncertainty exactly where the paper claims
none.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["RTResult", "surprisal_from_corpus", "reading_time_gain", "spillover"]


@dataclass
class RTResult:
    gain_per_word: float
    baseline_ll: float
    full_ll: float
    n_words: int
    n_folds: int
    estimator: str
    converged: bool
    note: str = ""


def surprisal_from_corpus(corpus, theta, model, kernel=None) -> np.ndarray:
    """Lossy-context surprisal of each target's own word under the fitted estimator.

    The target word is the corpus word, which is the candidate the cache was
    built around; a target whose word fell outside the candidate set is returned
    as ``nan`` and dropped downstream rather than given the bucket's surprisal.
    """
    from lcsa.kernels import POWER
    from lcsa.likelihood import evaluate_target

    kernel = POWER if kernel is None else kernel
    out = np.full(len(corpus), np.nan, dtype=np.float64)
    for t, tgt in enumerate(corpus):
        fit = evaluate_target(tgt, theta, model, corpus.M, kernel)
        idx = int(getattr(tgt, "target_slot", -1))
        if idx < 0 or idx >= tgt.V:
            continue
        out[t] = -float(np.log(max(fit.q[idx], 1e-300)))
    return out


def spillover(x: np.ndarray, passage: np.ndarray, lag: int = 1) -> np.ndarray:
    """Value from ``lag`` words earlier within the same passage, ``nan`` at the edge.

    Spillover is standard in reading-time models and is included because omitting
    it inflates the apparent contribution of the current word's surprisal.
    """
    x = np.asarray(x, dtype=np.float64)
    passage = np.asarray(passage)
    out = np.full(x.size, np.nan, dtype=np.float64)
    for p in np.unique(passage):
        idx = np.flatnonzero(passage == p)
        if idx.size > lag:
            out[idx[lag:]] = x[idx[:-lag]]
    return out


def _fit_ll(y, X, groups, use_mixed: bool):
    """Return ``(loglik_fn, estimator_name, converged)`` for a held-out evaluation."""
    import numpy as np

    if use_mixed:
        try:
            import statsmodels.api as sm

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md = sm.MixedLM(y, X, groups=groups)
                res = md.fit(method="lbfgs", maxiter=200)
            beta = np.asarray(res.fe_params, dtype=np.float64)
            sigma2 = float(res.scale)
            if np.isfinite(sigma2) and sigma2 > 0 and np.all(np.isfinite(beta)):
                return beta, sigma2, "MixedLM", bool(res.converged)
        except Exception as exc:  # noqa: BLE001 - reported, then degraded
            log.warning("MixedLM failed (%s: %s); falling back to OLS", type(exc).__name__, exc)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(X.shape[0] - X.shape[1], 1)
    sigma2 = float(resid @ resid / dof)
    return beta, max(sigma2, 1e-12), "OLS", True


def _gauss_ll(y, X, beta, sigma2):
    r = y - X @ beta
    return float(-0.5 * (np.log(2 * np.pi * sigma2) * y.size + (r @ r) / sigma2))


def reading_time_gain(
    gaze: np.ndarray,
    surprisal_full: np.ndarray,
    surprisal_lossy: np.ndarray,
    controls: np.ndarray,
    passage: np.ndarray,
    n_folds: int = 5,
    use_mixed: bool = True,
    seed: int = 0,
) -> RTResult:
    """Held-out per-word log-likelihood gain from adding the lossy-context term.

    Baseline predictors: the controls plus full-context surprisal and its
    one-word spillover.  The comparison model adds lossy-context surprisal and
    its spillover.  Folds partition passages, so no passage appears in both
    training and evaluation.
    """
    y = np.asarray(gaze, dtype=np.float64)
    passage = np.asarray(passage)
    base_cols = [
        np.ones(y.size),
        np.asarray(surprisal_full, dtype=np.float64),
        spillover(surprisal_full, passage),
    ]
    ctrl = np.atleast_2d(np.asarray(controls, dtype=np.float64))
    if ctrl.shape[0] != y.size:
        ctrl = ctrl.T
    base = np.column_stack(base_cols + [ctrl])
    extra = np.column_stack(
        [np.asarray(surprisal_lossy, dtype=np.float64),
         spillover(np.asarray(surprisal_lossy, dtype=np.float64), passage)]
    )
    full = np.column_stack([base, extra])

    ok = np.isfinite(y) & np.isfinite(base).all(axis=1) & np.isfinite(full).all(axis=1)
    y, base, full, passage = y[ok], base[ok], full[ok], passage[ok]
    if y.size < 20:
        return RTResult(float("nan"), float("nan"), float("nan"), int(y.size), 0,
                        "none", False, "fewer than 20 usable words")

    rng = np.random.default_rng(seed)
    uniq = np.unique(passage)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, min(n_folds, uniq.size))
    ll_b = ll_f = 0.0
    est, conv = "OLS", True
    for f in folds:
        te = np.isin(passage, f)
        tr = ~te
        if tr.sum() < base.shape[1] + 3 or te.sum() == 0:
            continue
        bb, sb, est, cb = _fit_ll(y[tr], base[tr], passage[tr], use_mixed)
        bf, sf, est, cf = _fit_ll(y[tr], full[tr], passage[tr], use_mixed)
        conv = conv and cb and cf
        ll_b += _gauss_ll(y[te], base[te], bb, sb)
        ll_f += _gauss_ll(y[te], full[te], bf, sf)
    n = int(y.size)
    return RTResult(
        gain_per_word=float((ll_f - ll_b) / n),
        baseline_ll=float(ll_b / n),
        full_ll=float(ll_f / n),
        n_words=n,
        n_folds=len(folds),
        estimator=est,
        converged=conv,
    )
