"""Cluster-robust efficient-score test, the naive likelihood ratio, and resampling.

The primary statistic is a score test at ``delta = 0`` with the nuisances at
their constrained MLE, clustered by passage, and built from the *efficient*
score

    s_c = sum_{t in c} [ s_delta - I_{delta phi} I_{phi phi}^{-1} s_phi ],
    T_CR = (sum_c s_c)^2 / [ (C/(C-1)) sum_c (s_c - sbar)^2 ]  ~  F(1, C-1).

Projecting the nuisance directions out leaves the numerator unchanged, because
the nuisance score sums vanish at the constrained MLE, and shrinks only the
denominator; the unprojected statistic is therefore conservative and is
returned beside the efficient one.

CR3 follows Bell and McCaffrey (2002) in its one-step jackknife form, with the
cluster leverage ``h_c = tr(I_{phi phi}^{-1} I_{phi phi, c})``.  A full refit
CR3 would cost one constrained fit per cluster per reader, roughly 22 CPU-hours
across the design, and the one-step form is what the paper reports.  The more
conservative of CR1 and CR3 gives the headline.

The naive likelihood ratio ``2(L(delta_hat) - L(0))`` against the
half-half mixture ``0.5 chi2_0 + 0.5 chi2_1`` is computed too, because the gap
between it and ``T_CR`` on the over-dispersed floor is registered prediction 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.stats import chi2, f as f_dist, norm

from lcsa.corpusdata import Corpus
from lcsa.kernels import POWER
from lcsa.fitting import FitResult, fit, fit_constrained
from lcsa.likelihood import Model, information, information_by_cluster, observed_scores

__all__ = [
    "ScoreTestResult",
    "score_test",
    "lr_test",
    "LRResult",
    "cluster_bootstrap",
    "tost",
    "TOSTResult",
    "rejection_rate",
]

_RIDGE = 1e-10


def _solve_psd(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``A x = b`` for a PSD ``A``, adding a proportional ridge if singular."""
    A = np.asarray(A, dtype=np.float64)
    scale = float(np.trace(A)) / max(A.shape[0], 1)
    ridge = _RIDGE * max(scale, 1.0)
    for _ in range(6):
        try:
            return np.linalg.solve(A + ridge * np.eye(A.shape[0]), b)
        except np.linalg.LinAlgError:
            ridge *= 100.0
    return np.linalg.lstsq(A, b, rcond=None)[0]


@dataclass
class ScoreTestResult:
    T_cr1: float
    p_cr1: float
    T_cr3: float
    p_cr3: float
    T_raw: float
    p_raw: float
    n_clusters: int
    s_cluster: np.ndarray
    numerator: float
    var_cr1: float
    var_cr3: float
    info_eff: float
    degenerate: bool
    null_fit: FitResult = field(repr=False, default=None)

    @property
    def design_effect(self) -> float:
        """Cluster-robust variance over the model-based variance of the same score.

        Under the multinomial model the variance of ``sum_c s_c`` is the
        efficient information, so this ratio is exactly the factor by which the
        naive likelihood ratio understates uncertainty.  It is 1 for the plain
        floor by construction and is the quantity the over-dispersed floor is
        calibrated to reproduce.
        """
        if self.info_eff <= 0:
            return float("nan")
        return float(self.var_cr1 / self.info_eff)

    @property
    def T(self) -> float:
        """Headline statistic: the more conservative of CR1 and CR3."""
        return min(self.T_cr1, self.T_cr3)

    @property
    def p(self) -> float:
        return max(self.p_cr1, self.p_cr3)

    def rejects(self, alpha: float = 0.05) -> bool:
        return self.p < alpha

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"ScoreTest(T={self.T:.3f}, p={self.p:.4f}, "
            f"CR1={self.T_cr1:.3f}/{self.p_cr1:.4f}, "
            f"CR3={self.T_cr3:.3f}/{self.p_cr3:.4f}, "
            f"deff={self.design_effect:.2f}, C={self.n_clusters})"
        )


def score_test(
    corpus: Corpus,
    model: Model,
    kernel=POWER,
    delta0: float = 0.0,
    null_fit: FitResult | None = None,
    n_starts: int = 3,
    seed: int = 0,
) -> ScoreTestResult:
    """Cluster-robust efficient-score test of ``delta = delta0``."""
    if null_fit is None:
        null_fit = fit_constrained(
            corpus, model, kernel, delta0=delta0, n_starts=n_starts, seed=seed
        )
    theta0 = null_fit.theta

    U = observed_scores(corpus, theta0, model, kernel)  # (T, P)
    I = information(corpus, theta0, model, kernel)
    C = corpus.n_clusters
    if C < 2:
        raise ValueError(f"cluster-robust inference needs at least 2 clusters, got {C}")

    I_dd = float(I[0, 0])
    I_dp = I[0, 1:]
    I_pp = I[1:, 1:]
    w = _solve_psd(I_pp, I_dp) if I_pp.size else np.zeros(0)

    eff = U[:, 0] - (U[:, 1:] @ w if w.size else 0.0)
    raw = U[:, 0]
    s_c = np.bincount(corpus.cluster_index, weights=eff, minlength=C)
    s_raw = np.bincount(corpus.cluster_index, weights=raw, minlength=C)

    num = float(s_c.sum()) ** 2
    var_cr1 = (C / (C - 1.0)) * float(((s_c - s_c.mean()) ** 2).sum())

    # CR3: one-step jackknife, leverage from the per-cluster information blocks.
    if I_pp.size:
        Ic = information_by_cluster(corpus, theta0, model, kernel)[:, 1:, 1:]
        Ipp_inv = np.linalg.pinv(I_pp + _RIDGE * max(float(np.trace(I_pp)) / I_pp.shape[0], 1.0) * np.eye(I_pp.shape[0]))
        h = np.array([float(np.trace(Ipp_inv @ Ic[c])) for c in range(C)])
    else:
        h = np.zeros(C)
    h = np.clip(h, 0.0, 1.0 - 1e-6)
    adj = s_c / (1.0 - h)
    var_cr3 = ((C - 1.0) / C) * float(((adj - adj.mean()) ** 2).sum())

    info_eff = I_dd - float(I_dp @ w) if w.size else I_dd

    var_raw = (C / (C - 1.0)) * float(((s_raw - s_raw.mean()) ** 2).sum())
    num_raw = float(s_raw.sum()) ** 2

    degenerate = not (var_cr1 > 0.0 and var_cr3 > 0.0)

    def _stat(n: float, v: float) -> tuple[float, float]:
        if not np.isfinite(v) or v <= 0.0:
            return 0.0, 1.0
        T = n / v
        return float(T), float(f_dist.sf(T, 1, C - 1))

    T1, p1 = _stat(num, var_cr1)
    T3, p3 = _stat(num, var_cr3)
    Tr, pr = _stat(num_raw, var_raw)
    return ScoreTestResult(
        T_cr1=T1, p_cr1=p1, T_cr3=T3, p_cr3=p3, T_raw=Tr, p_raw=pr,
        n_clusters=C, s_cluster=s_c, numerator=float(s_c.sum()),
        var_cr1=var_cr1, var_cr3=var_cr3, info_eff=float(max(info_eff, 0.0)),
        degenerate=degenerate, null_fit=null_fit,
    )


@dataclass
class LRResult:
    LR: float
    p: float
    delta_hat: float
    at_bound: bool
    full_fit: FitResult = field(repr=False, default=None)
    null_fit: FitResult = field(repr=False, default=None)

    def rejects(self, alpha: float = 0.05) -> bool:
        return self.p < alpha


def lr_test(
    corpus: Corpus,
    model: Model,
    kernel=POWER,
    full_fit: FitResult | None = None,
    null_fit: FitResult | None = None,
    n_starts: int = 3,
    seed: int = 0,
) -> LRResult:
    """Naive likelihood ratio against ``0.5 chi2_0 + 0.5 chi2_1``.

    Reported for every reader as the comparison the field currently uses, and
    never as this paper's primary statistic; the boundary mixture is the correct
    reference because ``delta >= 0`` puts the null on the edge of the parameter
    space (Self and Liang, 1987).
    """
    if full_fit is None:
        full_fit = fit(corpus, model, kernel, n_starts=n_starts, seed=seed)
    if null_fit is None:
        null_fit = fit_constrained(corpus, model, kernel, n_starts=n_starts, seed=seed)
    LR = 2.0 * (full_fit.loglik - null_fit.loglik)
    LR = max(LR, 0.0)
    p = 0.5 * float(chi2.sf(LR, 1)) if LR > 0 else 1.0
    return LRResult(
        LR=float(LR), p=p, delta_hat=full_fit.delta, at_bound=full_fit.at_bound,
        full_fit=full_fit, null_fit=null_fit,
    )


def cluster_bootstrap(
    corpus: Corpus,
    statistic: Callable[[Corpus], dict],
    n_boot: int = 200,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Paired cluster bootstrap: resample the 55 passages with replacement.

    Every reader is refitted on the *same* resample by the caller's ``statistic``
    closure, which is what makes the human-versus-null contrast paired.  A
    replicate whose fit lands on the boundary is kept and recorded as such; a
    replicate that raises is recorded with ``error`` rather than dropped, because
    silently dropping non-convergences would bias the distribution toward the
    paper's own prediction.
    """
    rng = np.random.default_rng(seed)
    C = corpus.n_clusters
    out: list[dict] = []
    for b in range(n_boot):
        draw = rng.integers(0, C, size=C)
        try:
            sub = corpus.subset_clusters(draw.tolist())
            rec = dict(statistic(sub))
            rec["error"] = None
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            rec = {"error": f"{type(exc).__name__}: {exc}"}
        rec["replicate"] = b
        out.append(rec)
        if progress is not None:
            progress(b + 1, n_boot)
    return out


@dataclass
class TOSTResult:
    margin: float
    diff: float
    se: float
    p_lower: float
    p_upper: float
    p: float
    equivalent: bool
    n_used: int


def tost(
    differences: Sequence[float],
    margin: float = 0.25,
    alpha: float = 0.05,
) -> TOSTResult:
    """Two one-sided tests for equivalence on a bootstrap difference distribution.

    ``differences`` are paired bootstrap replicates of ``log delta_human -
    log delta_null``.  Non-finite replicates (a boundary or unbounded fit on
    either side) are excluded from the moments and counted, since a log
    difference is undefined there; the count is returned so that an equivalence
    claim resting on few usable replicates is visible.
    """
    d = np.asarray([x for x in differences if np.isfinite(x)], dtype=np.float64)
    if d.size < 3:
        return TOSTResult(margin, float("nan"), float("nan"), 1.0, 1.0, 1.0, False, int(d.size))
    m = float(d.mean())
    se = float(d.std(ddof=1))
    if se <= 0.0:
        eq = abs(m) < margin
        return TOSTResult(margin, m, 0.0, 0.0 if eq else 1.0, 0.0 if eq else 1.0,
                          0.0 if eq else 1.0, eq, int(d.size))
    p_lo = float(norm.sf((m + margin) / se))
    p_hi = float(norm.cdf((m - margin) / se))
    p = max(p_lo, p_hi)
    return TOSTResult(margin, m, se, p_lo, p_hi, p, p < alpha, int(d.size))


def rejection_rate(p_values: Sequence[float], alpha: float = 0.05) -> tuple[float, float, float]:
    """``(rate, lo, hi)`` with a Wilson 95 percent interval on the rate."""
    p = np.asarray([x for x in p_values if np.isfinite(x)], dtype=np.float64)
    n = p.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    k = float((p < alpha).sum())
    rate = k / n
    z = 1.959963984540054
    den = 1.0 + z * z / n
    centre = (rate + z * z / (2 * n)) / den
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / den
    return float(rate), float(max(0.0, centre - half)), float(min(1.0, centre + half))
