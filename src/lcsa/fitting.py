"""Maximum likelihood, constrained fits, and profile-likelihood regions.

L-BFGS-B with the analytic gradient of :mod:`lcsa.likelihood`, box constraints
from ``Model.bounds`` and multiple starts.  Two behaviours matter for the
paper's honesty and are implemented deliberately rather than incidentally.

First, a fit that lands on the ``delta`` upper bound is reported as *at bound*
rather than silently as a point estimate, because a boundary fit is the visible
form of an unidentified direction and dropping it would bias the bootstrap
toward the paper's own prediction.

Second, the profile region is computed on a fixed log-spaced grid with every
other parameter re-optimised at each point, and a region that reaches either end
of the grid is returned with an explicit ``unbounded`` flag.  Nothing downstream
converts an unbounded region into a number.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from lcsa.corpusdata import Corpus
from lcsa.kernels import POWER
from lcsa.likelihood import Model, loglik_and_grad

__all__ = ["FitResult", "fit", "fit_constrained", "profile_curve", "profile_interval",
           "local_grid",
           "ProfileRegion"]

_DELTA_MAX = 5.0
_BOUND_TOL = 1e-4


@dataclass
class FitResult:
    theta: np.ndarray
    loglik: float
    success: bool
    message: str
    n_eval: int
    at_bound: bool
    model_name: str
    param_names: list[str] = field(default_factory=list)

    @property
    def delta(self) -> float:
        return float(self.theta[0])

    def as_dict(self) -> dict:
        d = {n: float(v) for n, v in zip(self.param_names, self.theta)}
        d.update(
            loglik=self.loglik,
            success=self.success,
            at_bound=self.at_bound,
            model=self.model_name,
        )
        return d


def _objective(corpus, model, kernel, free_idx, fixed_vals, template):
    def fun(x):
        theta = template.copy()
        theta[free_idx] = x
        try:
            L, g = loglik_and_grad(corpus, theta, model, kernel)
        except (FloatingPointError, ValueError):
            # An interior point should never produce this, but L-BFGS-B probes
            # the boundary; returning a large finite value keeps the line search
            # alive instead of aborting the whole fit.
            return 1e30, np.zeros(len(free_idx))
        return -L, -g[free_idx]

    return fun


def _starts(model: Model, M: int, n_starts: int, seed: int) -> list[np.ndarray]:
    base = model.start(M)
    out = [base.copy()]
    if n_starts > 1:
        alt = base.copy()
        alt[0], alt[1], alt[2] = 0.30, 0.15, 0.85
        out.append(alt)
    if n_starts > 2:
        alt = base.copy()
        alt[0], alt[1], alt[2] = 0.05, 0.02, 1.20
        out.append(alt)
    rng = np.random.default_rng(seed)
    while len(out) < n_starts:
        x = base.copy()
        x[0] = float(rng.uniform(0.0, 0.8))
        x[1] = float(rng.uniform(0.01, 0.30))
        x[2] = float(rng.uniform(0.6, 1.4))
        if len(x) > 3:
            x[3:] = rng.normal(scale=0.05, size=len(x) - 3)
        out.append(x)
    return out


def fit(
    corpus: Corpus,
    model: Model,
    kernel=POWER,
    fixed: dict[int, float] | None = None,
    n_starts: int = 3,
    seed: int = 0,
    delta_max: float = _DELTA_MAX,
    maxiter: int = 500,
    start: np.ndarray | None = None,
) -> FitResult:
    """Maximise the log-likelihood, optionally with some coordinates pinned.

    ``fixed`` maps parameter index to value; index 0 is ``delta``, so
    ``fixed={0: 0.0}`` gives the constrained fit the score test needs.
    """
    M = corpus.M
    P = model.dim(M)
    fixed = dict(fixed or {})
    for i in fixed:
        if not 0 <= i < P:
            raise ValueError(f"fixed index {i} out of range for a {P}-parameter model")
    free_idx = np.array([i for i in range(P) if i not in fixed], dtype=int)
    if free_idx.size == 0:
        theta = np.array([fixed[i] for i in range(P)], dtype=np.float64)
        L, _ = loglik_and_grad(corpus, theta, model, kernel)
        return FitResult(theta, float(L), True, "all parameters fixed", 0, False,
                         model.name, model.param_names(M, corpus.feature_names))

    bounds_all = model.bounds(M, delta_max)
    bounds = [bounds_all[i] for i in free_idx]

    starts = _starts(model, M, n_starts, seed)
    if start is not None:
        # A warm start from a neighbouring fit is tried first and kept only if it
        # wins, so it can speed the profile up without ever making it worse.
        w = np.asarray(start, dtype=np.float64).copy()
        if w.size != P:
            raise ValueError(f"start has {w.size} entries, expected {P}")
        starts = [w] + starts

    best = None
    n_eval = 0
    for x0 in starts:
        template = x0.copy()
        for i, v in fixed.items():
            template[i] = v
        x0f = np.clip(
            template[free_idx],
            [b[0] for b in bounds],
            [b[1] for b in bounds],
        )
        fun = _objective(corpus, model, kernel, free_idx, fixed, template)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                fun,
                x0f,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
            )
        n_eval += int(res.nfev)
        theta = template.copy()
        theta[free_idx] = res.x
        L = -float(res.fun)
        if not np.isfinite(L):
            continue
        if best is None or L > best[1]:
            best = (theta, L, bool(res.success), str(res.message))

    if best is None:
        raise RuntimeError(
            "every start failed to produce a finite log-likelihood; "
            "check the cache for zero rows or a mismatched candidate set"
        )
    theta, L, ok, msg = best
    at_bound = (0 not in fixed) and (theta[0] >= delta_max - _BOUND_TOL)
    return FitResult(
        theta=theta,
        loglik=L,
        success=ok,
        message=msg,
        n_eval=n_eval,
        at_bound=at_bound,
        model_name=model.name,
        param_names=model.param_names(M, corpus.feature_names),
    )


def fit_constrained(corpus: Corpus, model: Model, kernel=POWER, delta0: float = 0.0,
                    **kw) -> FitResult:
    """Fit with ``delta`` pinned at ``delta0``; the null fit for the score test."""
    return fit(corpus, model, kernel, fixed={0: float(delta0)}, **kw)


@dataclass
class ProfileRegion:
    lo: float
    hi: float
    unbounded_lo: bool
    unbounded_hi: bool
    grid: np.ndarray
    profile: np.ndarray
    max_loglik: float

    @property
    def unbounded(self) -> bool:
        return self.unbounded_lo or self.unbounded_hi

    def __repr__(self) -> str:  # pragma: no cover - display only
        lo = "unbounded" if self.unbounded_lo else f"{self.lo:.4f}"
        hi = "unbounded" if self.unbounded_hi else f"{self.hi:.4f}"
        return f"ProfileRegion({lo}, {hi})"


def default_grid(n: int = 21, lo: float = 1e-3, hi: float = 2.0) -> np.ndarray:
    """21 points log-spaced in ``delta``, with an exact zero prepended.

    Zero is the hypothesis under test and log spacing cannot reach it, so it is
    carried as a separate grid point rather than approximated by ``1e-3``.
    """
    return np.concatenate([[0.0], np.geomspace(lo, hi, n - 1)])


def local_grid(delta_hat: float, se: float | None = None, n: int = 9,
               k: float = 6.0, span: float = 4.0) -> np.ndarray:
    """A short grid around ``delta_hat``, with an exact zero prepended.

    Coverage replicates cannot afford 21 constrained fits each, and a fixed grid
    spanning three orders of magnitude puts at most two points anywhere near the
    optimum, so its interpolated endpoints are far too tight.  Given the
    cluster-corrected standard error the grid runs ``k`` standard errors either
    side of the estimate and always contains it, which is where the deviance
    actually crosses the threshold.  Without one it falls back to a
    multiplicative span, and with no usable estimate at all to the default grid.
    """
    d = float(delta_hat)
    if not np.isfinite(d) or d < 0:
        return default_grid(n=n)
    m = max(int(n) - 1, 3)
    if se is not None and np.isfinite(se) and se > 0:
        lo = max(0.0, d - k * float(se))
        hi = d + k * float(se)
        g = np.linspace(lo, hi, m)
    elif d > 0:
        g = np.geomspace(d / span, d * span, m)
    else:
        return default_grid(n=n)
    g = np.unique(np.concatenate(([0.0, d], g)))
    return g


def profile_curve(
    corpus: Corpus,
    model: Model,
    kernel=POWER,
    grid: np.ndarray | None = None,
    n_starts: int = 2,
    seed: int = 0,
    warm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Profile log-likelihood over ``delta``: ``(grid, values)``.

    Each constrained fit is warm-started from the previous grid point's solution,
    and from ``warm`` (normally the unconstrained MLE) at the first point.  This
    is not only faster: a cold constrained fit that stops short of its optimum
    lowers the profile, which narrows the region and can exclude the MLE itself,
    and that failure is invisible unless you look for it.
    """
    g = default_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    vals = np.empty(g.size, dtype=np.float64)
    prev = None if warm is None else np.asarray(warm, dtype=np.float64).copy()
    for i, d in enumerate(g):
        f = fit(corpus, model, kernel, fixed={0: float(d)}, n_starts=n_starts,
                seed=seed, start=prev)
        vals[i] = f.loglik
        prev = f.theta.copy()
    return g, vals


def profile_interval(
    corpus: Corpus,
    model: Model,
    kernel=POWER,
    grid: np.ndarray | None = None,
    level: float = 0.95,
    scale: float = 1.0,
    n_starts: int = 2,
    seed: int = 0,
    curve: tuple[np.ndarray, np.ndarray] | None = None,
    max_loglik: float | None = None,
    warm: np.ndarray | None = None,
) -> ProfileRegion:
    """Profile region ``{delta : 2 scale (Lmax - Lprof) <= chi2_1(level)}``.

    ``scale`` carries the cluster-robust correction: with 55 clusters the naive
    profile is too narrow by the same factor the score test's denominator
    corrects, and passing ``scale < 1`` widens the region accordingly.  Interval
    endpoints are found by linear interpolation of the deviance between grid
    points, and a region touching either end of the grid is flagged rather than
    truncated.
    """
    if curve is None:
        g, vals = profile_curve(corpus, model, kernel, grid, n_starts, seed, warm=warm)
    else:
        g, vals = curve
    g = np.asarray(g, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    cut = chi2.ppf(level, 1)
    # The reference is the unconstrained maximum when the caller has it: on a
    # coarse grid the grid maximum sits below it, and anchoring there shifts the
    # whole region toward the nearest grid point.
    Lmax = float(vals.max()) if max_loglik is None else max(float(max_loglik), float(vals.max()))
    dev = 2.0 * float(scale) * (Lmax - vals)
    inside = dev <= cut
    if not inside.any():
        # Numerically possible only if the maximum itself is excluded, which
        # cannot happen since dev == 0 there; guard anyway.
        raise RuntimeError("empty profile region; the deviance curve is degenerate")
    idx = np.flatnonzero(inside)
    i0, i1 = int(idx[0]), int(idx[-1])

    def cross(a: int, b: int) -> float:
        """Where the deviance crosses ``cut`` between grid points ``a`` and ``b``.

        A profile deviance is close to quadratic, so linear interpolation across
        a wide log-spaced gap systematically places the crossing too close to the
        maximum and produces regions that are too narrow.  A parabola through the
        crossing pair plus one outer neighbour fixes that; the linear form is the
        fallback when no third point exists or the parabola misbehaves.
        """
        da, db = dev[a], dev[b]
        if db == da:
            return float(g[b])
        lin = float(g[a] + (cut - da) / (db - da) * (g[b] - g[a]))
        c = a - (b - a)  # the point on the far side of ``a``
        if 0 <= c < g.size:
            xs = np.array([g[c], g[a], g[b]], dtype=np.float64)
            ys = np.array([dev[c], dev[a], dev[b]], dtype=np.float64)
            try:
                q = np.polyfit(xs, ys, 2)
            except Exception:
                return lin
            roots = np.roots([q[0], q[1], q[2] - cut])
            lo_x, hi_x = min(g[a], g[b]), max(g[a], g[b])
            real = [float(r.real) for r in roots
                    if abs(r.imag) < 1e-9 and lo_x - 1e-12 <= r.real <= hi_x + 1e-12]
            if real:
                return min(real, key=lambda r: abs(r - lin))
        return lin

    lo = float(g[i0]) if i0 == 0 else cross(i0, i0 - 1)
    hi = float(g[i1]) if i1 == g.size - 1 else cross(i1, i1 + 1)
    return ProfileRegion(
        lo=lo,
        hi=hi,
        unbounded_lo=i0 == 0 and g[0] > 0.0,
        unbounded_hi=i1 == g.size - 1,
        grid=g,
        profile=vals,
        max_loglik=float(vals.max()),
    )
