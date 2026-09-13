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

Four denominators are reported.  CR1 is the plain cluster sandwich with the
``C/(C-1)`` factor and an ``F(1, C-1)`` reference.  CR2 and CR3 follow Bell
and McCaffrey (2002): the cluster leverage is the share of the *efficient*
information carried by the cluster, ``h_c = I_eff,c / I_eff``, which is the
``(I^{-1} I_c I^{-1})_{dd} / (I^{-1})_{dd}`` form written out, and it sums to
one over clusters.  CR2 scales ``s_c`` by ``(1 - h_c)^{-1/2}`` and is referred
to a ``t`` distribution with the Satterthwaite degrees of freedom of Bell and
McCaffrey, ``tr(M Sigma)^2 / tr((M Sigma)^2)`` with ``Sigma = diag(I_eff,c)``;
the cross term from fitting the nuisances is ignored in ``Sigma`` because it
sums to zero over clusters.  CR3 scales by ``(1 - h_c)^{-1}``.  A full refit
CR3 would cost one constrained fit per cluster per reader and is not run.

The headline p-value is a wild cluster bootstrap of the efficient score in the
style of Kline and Santos (2012): the cluster sums ``s_c`` are sign-flipped
with Rademacher weights, the CR1 statistic is recomputed for each draw without
any refit, and the p-value is the share of draws at least as extreme as the
observed statistic (all ``2^{C-1}`` sign patterns are enumerated when that is
cheaper than the requested number of draws).  The test is one-sided towards
positive decay, through the signed root ``z = sign(sum s_c) sqrt(T)``, because
the naive likelihood ratio it is compared with tests the same one-sided
boundary hypothesis; two-sided values are kept beside it.

The naive likelihood ratio ``2(L(delta_hat) - L(0))`` against the
half-half mixture ``0.5 chi2_0 + 0.5 chi2_1`` is computed too, because the gap
between it and ``T_CR`` on the over-dispersed floor is registered prediction 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.stats import chi2, f as f_dist, norm, t as t_dist

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
    T_cr2: float = float("nan")
    p_cr2: float = float("nan")
    var_cr2: float = float("nan")
    df_bm: float = float("nan")
    leverage: np.ndarray = field(repr=False, default=None)
    z: float = float("nan")
    p_one_cr1: float = float("nan")
    p_one_cr2: float = float("nan")
    p_one_cr3: float = float("nan")
    p_wild: float = float("nan")
    p_wild_one: float = float("nan")
    n_wild: int = 0
    wild_enumerated: bool = False

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
        """Headline statistic: the CR1 form that the wild bootstrap resamples."""
        return self.T_cr1

    @property
    def p(self) -> float:
        """Headline p-value: one-sided wild cluster bootstrap of the efficient score.

        Falls back to the one-sided CR2 ``t`` value when no bootstrap draws
        were requested, and to CR1 when the leverage is unavailable.
        """
        for v in (self.p_wild_one, self.p_one_cr2, self.p_one_cr1):
            if np.isfinite(v):
                return float(v)
        return float(self.p_cr1)

    def rejects(self, alpha: float = 0.05) -> bool:
        return self.p < alpha

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"ScoreTest(T={self.T:.3f}, p={self.p:.4f}, "
            f"CR1={self.T_cr1:.3f}/{self.p_cr1:.4f}, "
            f"CR2={self.T_cr2:.3f}/{self.p_cr2:.4f}/df={self.df_bm:.1f}, "
            f"CR3={self.T_cr3:.3f}/{self.p_cr3:.4f}, wild={self.p_wild_one:.4f}, "
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
    n_wild: int = 999,
) -> ScoreTestResult:
    """Cluster-robust efficient-score test of ``delta = delta0``.

    ``n_wild`` is the number of Rademacher draws of the wild cluster bootstrap;
    zero skips it, which the calibration loops use because the CR forms are all
    they need.
    """
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

    # Leverage: the share of the efficient information carried by each cluster.
    Ic = information_by_cluster(corpus, theta0, model, kernel)  # (C, P, P)
    g = np.concatenate([[1.0], -w]) if w.size else np.ones(1)
    info_eff_c = np.einsum("i,cij,j->c", g, Ic, g)
    info_eff = float(max(I_dd - float(I_dp @ w) if w.size else I_dd, 0.0))
    h = info_eff_c / info_eff if info_eff > 0 else np.zeros(C)
    h = np.clip(h, 0.0, 1.0 - 1e-6)

    adj2 = s_c / np.sqrt(1.0 - h)
    var_cr2 = float(((adj2 - adj2.mean()) ** 2).sum())
    adj3 = s_c / (1.0 - h)
    var_cr3 = ((C - 1.0) / C) * float(((adj3 - adj3.mean()) ** 2).sum())
    df_bm = _satterthwaite_df(h, np.clip(info_eff_c, 0.0, None))

    var_raw = (C / (C - 1.0)) * float(((s_raw - s_raw.mean()) ** 2).sum())
    num_raw = float(s_raw.sum()) ** 2

    degenerate = not (var_cr1 > 0.0 and var_cr2 > 0.0 and var_cr3 > 0.0)

    def _stat(n: float, v: float, df: float) -> tuple[float, float]:
        if not np.isfinite(v) or v <= 0.0 or not np.isfinite(df) or df <= 0:
            return 0.0, 1.0
        T = n / v
        return float(T), float(f_dist.sf(T, 1, df))

    T1, p1 = _stat(num, var_cr1, C - 1.0)
    T2, p2 = _stat(num, var_cr2, df_bm)
    T3, p3 = _stat(num, var_cr3, C - 1.0)
    Tr, pr = _stat(num_raw, var_raw, C - 1.0)

    sgn = float(np.sign(s_c.sum())) or 1.0
    z = sgn * np.sqrt(T1)

    def _one(T: float, df: float) -> float:
        if T <= 0.0 or not np.isfinite(df) or df <= 0:
            return 1.0
        return float(t_dist.sf(sgn * np.sqrt(T), df))

    p1_one, p2_one, p3_one = _one(T1, C - 1.0), _one(T2, df_bm), _one(T3, C - 1.0)

    p_wild = p_wild_one = float("nan")
    enumerated = False
    if n_wild > 0 and var_cr1 > 0.0:
        p_wild, p_wild_one, enumerated = _wild_score_bootstrap(s_c, T1, z, n_wild, seed)

    return ScoreTestResult(
        T_cr1=T1, p_cr1=p1, T_cr3=T3, p_cr3=p3, T_raw=Tr, p_raw=pr,
        n_clusters=C, s_cluster=s_c, numerator=float(s_c.sum()),
        var_cr1=var_cr1, var_cr3=var_cr3, info_eff=info_eff,
        degenerate=degenerate, null_fit=null_fit,
        T_cr2=T2, p_cr2=p2, var_cr2=var_cr2, df_bm=float(df_bm), leverage=h,
        z=float(z), p_one_cr1=p1_one, p_one_cr2=p2_one, p_one_cr3=p3_one,
        p_wild=p_wild, p_wild_one=p_wild_one, n_wild=int(2 ** (C - 1) if enumerated else n_wild),
        wild_enumerated=enumerated,
    )


def _satterthwaite_df(h: np.ndarray, sigma2: np.ndarray) -> float:
    """Bell and McCaffrey degrees of freedom for the CR2 form of a cluster-sum test.

    The statistic's denominator is ``s' M s`` with ``M = D' A^2 D``, ``D`` the
    centring matrix and ``A = diag((1 - h_c)^{-1/2})``; under a working model
    with ``Cov(s) = Sigma = diag(sigma2)`` the two-moment match gives
    ``df = tr(M Sigma)^2 / tr((M Sigma)^2)``.
    """
    C = h.size
    if C < 2:
        return float("nan")
    D = np.eye(C) - np.full((C, C), 1.0 / C)
    A2 = np.diag(1.0 / (1.0 - h))
    M = D @ A2 @ D
    MS = M * sigma2[None, :]  # M @ diag(sigma2)
    tr1 = float(np.trace(MS))
    tr2 = float(np.sum(MS * MS.T))  # tr((MS)^2)
    if tr2 <= 0.0 or not np.isfinite(tr2):
        return float("nan")
    return float(tr1 * tr1 / tr2)


def _wild_score_bootstrap(s_c: np.ndarray, T_obs: float, z_obs: float,
                          n_wild: int, seed: int) -> tuple[float, float, bool]:
    """Rademacher sign-flip bootstrap of the CR1 statistic, no refits.

    Returns the two-sided p (share of ``T*`` at least ``T_obs``), the one-sided
    p (share of signed roots ``z*`` at least ``z_obs``) and whether the sign
    patterns were enumerated exactly.  Random draws use the ``(1 + k)/(1 + n)``
    convention so that the smallest attainable p is not zero.
    """
    C = s_c.size
    if 2 ** (C - 1) <= n_wild:
        # All patterns up to a global sign, which leaves T* unchanged.
        bits = ((np.arange(2 ** (C - 1))[:, None] >> np.arange(C - 1)[None, :]) & 1)
        eps = np.concatenate([np.ones((bits.shape[0], 1)), 1.0 - 2.0 * bits], axis=1)
        enumerated = True
    else:
        rng = np.random.default_rng([int(seed), 7919])
        eps = rng.choice([-1.0, 1.0], size=(n_wild, C))
        enumerated = False
    S = eps * s_c[None, :]
    tot = S.sum(axis=1)
    var = (C / (C - 1.0)) * ((S - S.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    ok = var > 0
    T_star = np.where(ok, tot ** 2 / np.where(ok, var, 1.0), 0.0)
    z_star = np.sign(tot) * np.sqrt(T_star)
    tol = 1e-9
    if enumerated:
        p_two = float(np.mean(T_star >= T_obs - tol))
        # Signed roots come in +/- pairs under the global sign, so the
        # one-sided value uses both signs of every enumerated pattern.
        p_one = float(0.5 * (np.mean(z_star >= z_obs - tol) + np.mean(-z_star >= z_obs - tol)))
    else:
        p_two = float((1 + np.sum(T_star >= T_obs - tol)) / (1 + T_star.size))
        p_one = float((1 + np.sum(z_star >= z_obs - tol)) / (1 + z_star.size))
    return p_two, p_one, enumerated


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
    reps=None,
) -> list[dict]:
    """Paired cluster bootstrap: resample the 55 passages with replacement.

    Every reader is refitted on the *same* resample by the caller's ``statistic``
    closure, which is what makes the human-versus-null contrast paired.  A
    replicate whose fit lands on the boundary is kept and recorded as such; a
    replicate that raises is recorded with ``error`` rather than dropped, because
    silently dropping non-convergences would bias the distribution toward the
    paper's own prediction.  Replicate ``b`` draws from a generator seeded by
    ``(seed, b)`` alone, so ``reps`` can be any range of absolute indices and
    disjoint ranges concatenate into the full run.
    """
    C = corpus.n_clusters
    reps = list(range(n_boot) if reps is None else reps)
    out: list[dict] = []
    for i, b in enumerate(reps):
        draw = np.random.default_rng([int(seed), int(b)]).integers(0, C, size=C)
        try:
            sub = corpus.subset_clusters(draw.tolist())
            rec = dict(statistic(sub))
            rec["error"] = None
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            rec = {"error": f"{type(exc).__name__}: {exc}"}
        rec["replicate"] = int(b)
        out.append(rec)
        if progress is not None:
            progress(i + 1, len(reps))
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
    df: float = float("inf")


def tost(
    differences: Sequence[float],
    margin: float = 0.25,
    alpha: float = 0.05,
    df: float | None = None,
) -> TOSTResult:
    """Two one-sided tests for equivalence on a bootstrap difference distribution.

    ``differences`` are paired bootstrap replicates of ``log delta_human -
    log delta_null``.  Non-finite replicates (a boundary or unbounded fit on
    either side) are excluded from the moments and counted, since a log
    difference is undefined there, and the count is returned so that an
    equivalence claim resting on few usable replicates is visible.

    ``df`` sets the reference distribution.  The caller passes the cluster count
    less one, which is the same degrees of freedom every other interval in the
    paper carries, because a normal reference on 55 clusters makes equivalence
    easier to declare and equivalence is the direction the registered
    prediction wants.  Leaving it ``None`` keeps the normal reference.
    """
    d = np.asarray([x for x in differences if np.isfinite(x)], dtype=np.float64)
    nu = float("inf") if df is None else float(df)
    if nu <= 0.0:
        raise ValueError(f"df must be positive, got {df!r}")
    ref = norm if df is None else t_dist(nu)
    if d.size < 3:
        return TOSTResult(margin, float("nan"), float("nan"), 1.0, 1.0, 1.0, False,
                          int(d.size), nu)
    m = float(d.mean())
    se = float(d.std(ddof=1))
    if se <= 0.0:
        eq = abs(m) < margin
        return TOSTResult(margin, m, 0.0, 0.0 if eq else 1.0, 0.0 if eq else 1.0,
                          0.0 if eq else 1.0, eq, int(d.size), nu)
    p_lo = float(ref.sf((m + margin) / se))
    p_hi = float(ref.cdf((m - margin) / se))
    p = max(p_lo, p_hi)
    return TOSTResult(margin, m, se, p_lo, p_hi, p, p < alpha, int(d.size), nu)


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
