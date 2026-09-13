"""The nuisance span B_t and the residual fractions the absorption criterion needs.

All geometry happens in the *whitened* coordinates ``x = sqrt(q) a``, because
the inner product that governs absorption is

    <a, b>_q = sum_w q(w) a(w) b(w) = x . y.

Whitening turns a weighted QR into an ordinary thin QR, keeps the projection
numerically stable when some ``q(w)`` is tiny, and never requires dividing by
``sqrt(q)``.  Every quantity the paper reports from this module is a norm ratio
or a covariance, and both are inner products, so nothing needs unwhitening.

The span itself is the span of the nuisance score directions at the fitting
point, dimension 2 for the naive estimator and 7 for the repaired one, not a
list of confounds chosen by hand.  Rank-deficient spans are handled by dropping
columns whose R diagonal falls below a relative tolerance, which happens for
real when a feature column is constant on a small candidate set.

Two projections live here and they answer different questions.  The
*per-target* projection onto ``B_t`` lets every context choose its own
nuisance coefficients; it is what the nulls are orthogonalised against and it
is a lower bound on what survives.  The nuisance vector is one global ``phi``,
so what the estimator can absorb is the *global tangent space*

    T = { h : h_t = B_t a for one a shared by every target },

and the bias of ``delta`` is governed by the residual of ``h`` after projecting
onto ``T`` under the count-weighted inner product ``<a, b>_N = sum_t N_t
<a_t, b_t>_{q_t}``.  The Gram matrix of ``T`` in these coordinates is the
nuisance information block ``I_pp``, so the global projection is one linear
solve, and its residual is orthogonal to the efficient score direction.  The
per-target residual fraction is never larger than the global one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lcsa.corpusdata import Corpus, Target
from lcsa.kernels import POWER
from lcsa.likelihood import Model, evaluate_target

__all__ = [
    "whiten",
    "centre",
    "orthonormal_span",
    "decompose",
    "residual_fraction",
    "corpus_residual_fraction",
    "human_residual",
    "ResidualReport",
    "GlobalResidual",
    "global_residual",
    "split_half_residual",
    "implied_bias",
    "lambda_curvature_leak",
    "explained_share",
]

_RANK_TOL = 1e-10


def centre(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Remove the ``q``-mean, so the vector lives in the tangent space of the simplex."""
    a = np.asarray(a, dtype=np.float64)
    return a - (q @ a) if a.ndim == 1 else a - (q @ a)[None, :]


def whiten(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``sqrt(q) * a`` along the candidate axis (axis 0 for matrices)."""
    s = np.sqrt(np.asarray(q, dtype=np.float64))
    a = np.asarray(a, dtype=np.float64)
    return a * s if a.ndim == 1 else a * s[:, None]


def orthonormal_span(B_raw: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Orthonormal basis of ``span(B_raw)`` in whitened coordinates, shape (V, k).

    ``B_raw`` is (V, m) of already-centred directions.  Columns are dropped in
    order of decreasing R-diagonal magnitude until the remaining ones are
    numerically independent, so a duplicated or constant nuisance direction
    reduces ``k`` instead of producing a singular projection.
    """
    X = whiten(np.asarray(B_raw, dtype=np.float64), q)
    if X.ndim != 2:
        raise ValueError(f"B_raw must be 2-D (V, m), got shape {X.shape}")
    if X.shape[1] == 0:
        return np.zeros((X.shape[0], 0), dtype=np.float64)
    Q, R = np.linalg.qr(X, mode="reduced")
    d = np.abs(np.diag(R))
    if d.size == 0 or d.max() == 0.0:
        return np.zeros((X.shape[0], 0), dtype=np.float64)
    keep = d > _RANK_TOL * d.max()
    return Q[:, keep]


def decompose(h: np.ndarray, B_raw: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split centred ``h`` into ``(h_B, h_perp)`` in whitened coordinates."""
    q = np.asarray(q, dtype=np.float64)
    x = whiten(centre(h, q), q)
    Qb = orthonormal_span(B_raw, q)
    if Qb.shape[1] == 0:
        return np.zeros_like(x), x
    par = Qb @ (Qb.T @ x)
    return par, x - par


def residual_fraction(h: np.ndarray, B_raw: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """``(||h_perp||/||h||, ||h||)`` under the ``q`` inner product.

    Returns a fraction of ``nan`` when ``||h|| == 0``, since a responder with no
    mismatch has no residual to report and reporting 0 would read as full
    absorption.
    """
    par, perp = decompose(h, B_raw, q)
    total = float(np.linalg.norm(par + perp))
    if total <= 0.0:
        return float("nan"), 0.0
    return float(np.linalg.norm(perp)) / total, total


@dataclass
class ResidualReport:
    """Corpus-level residual summary, response-count weighted."""

    fraction: float
    fraction_unweighted: float
    span_dim_mean: float
    n_targets_used: int
    per_target: np.ndarray

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"ResidualReport(fraction={self.fraction:.4f}, "
            f"unweighted={self.fraction_unweighted:.4f}, "
            f"span_dim={self.span_dim_mean:.2f}, n={self.n_targets_used})"
        )


def human_residual(
    tgt: Target, q: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """``h = log(smoothed empirical) - log q_t``, centred under ``q``.

    ``q`` is the fitted null distribution of the target, the point the
    expansion of Proposition 2 is taken around; measuring the mismatch against
    the empty-context row of the cache, as an earlier version did, would put
    the whole context effect into ``h``.  Jeffreys smoothing with ``alpha =
    0.5`` keeps ``h`` finite when a candidate drew zero responses, which is the
    common case on a 120-type candidate set with 40 responses.  The
    sensitivity of every reported fraction to ``alpha in {0.1, 0.5, 1.0}`` is
    recorded beside it, because the smoothing bias is common to both halves of
    a split and the split-half debiasing cannot remove it.
    """
    n = np.asarray(tgt.n, dtype=np.float64)
    p_emp = (n + alpha) / (n.sum() + alpha * n.size)
    h = np.log(p_emp) - np.log(np.clip(q, 1e-300, None))
    return centre(h, q)


def corpus_residual_fraction(
    corpus: Corpus,
    theta: np.ndarray,
    model: Model,
    kernel=POWER,
    alpha: float = 0.5,
    h_fn=None,
) -> ResidualReport:
    """Per-target residual fraction of the mismatch against the nuisance span.

    Every target projects onto its own ``B_t`` with its own coefficients, so
    this is what a *target-specific* nuisance vector could absorb and it is a
    lower bound on the global fraction that :func:`global_residual` reports;
    the quantity that enters the bias expression of Proposition 2 is the
    global one.  The headline number is the response-count-weighted root of
    the ratio of summed squared norms; the unweighted mean over targets is
    reported beside it so that a handful of high-count passages cannot drive
    the fraction alone.
    """
    M = corpus.M
    num_w = den_w = 0.0
    fracs, dims, weights = [], [], []
    for tgt in corpus:
        if tgt.N <= 0:
            continue
        fit = evaluate_target(tgt, theta, model, M, kernel)
        h = h_fn(tgt, fit.q) if h_fn is not None else human_residual(tgt, fit.q, alpha)
        B = fit.scores[:, 1:]
        par, perp = decompose(h, B, fit.q)
        tot2 = float(par @ par + perp @ perp)
        per2 = float(perp @ perp)
        if tot2 <= 0.0:
            continue
        num_w += tgt.N * per2
        den_w += tgt.N * tot2
        fracs.append(np.sqrt(per2 / tot2))
        dims.append(orthonormal_span(B, fit.q).shape[1])
        weights.append(tgt.N)
    if not fracs:
        raise ValueError("no target contributed a non-degenerate residual")
    return ResidualReport(
        fraction=float(np.sqrt(num_w / den_w)),
        fraction_unweighted=float(np.mean(fracs)),
        span_dim_mean=float(np.mean(dims)),
        n_targets_used=len(fracs),
        per_target=np.asarray(fracs, dtype=np.float64),
    )


def orthogonalise_against_span(
    corpus: Corpus,
    h_list: list[np.ndarray],
    theta: np.ndarray,
    model: Model,
    kernel=POWER,
) -> list[np.ndarray]:
    """Return each ``h_t`` with its ``B_t`` component removed, in *unwhitened* units.

    This builds the substantive nulls: their tilt must be orthogonal to the
    measured nuisance span, so that the rejection they produce is attributable
    to the residual direction and not to something the nuisances could express.
    Dividing by ``sqrt(q)`` is safe here because the result is used only as an
    exponential tilt, and any coordinate with negligible ``q`` contributes
    negligibly to the resulting distribution; coordinates below the floor are
    set to zero rather than amplified.
    """
    M = corpus.M
    out = []
    for t, tgt in enumerate(corpus):
        fit = evaluate_target(tgt, theta, model, M, kernel)
        _, perp = decompose(h_list[t], fit.scores[:, 1:], fit.q)
        s = np.sqrt(fit.q)
        safe = s > 1e-8
        h = np.zeros_like(perp)
        h[safe] = perp[safe] / s[safe]
        out.append(h - fit.q @ h)
    return out


# -- the global tangent space -------------------------------------------------


@dataclass
class GlobalResidual:
    """The mismatch ``h`` against the global tangent space ``T``.

    ``fraction`` is ``||h_perp||_N / ||h||_N`` with ``h_perp`` the residual
    after the best *single* nuisance coefficient vector; ``fraction_per_target``
    is the older per-target number, a lower bound.  ``alignment`` is the cosine
    between ``h_perp`` and the efficient score direction under ``<., .>_N``,
    which is what turns a residual into bias: a large residual at right angles
    to the efficient score biases nothing.  ``coefficients`` are the projection
    coefficients ``a = I_pp^{-1} sum_t N_t B_t^T h_t`` in the nuisance order of
    the estimator, and ``inner_eff`` is ``<h, psi_eff>_N``, the numerator of
    the implied first-order bias.
    """

    fraction: float
    fraction_per_target: float
    alignment: float
    inner_eff: float
    info_eff: float
    norm2_total: float
    norm2_perp: float
    coefficients: np.ndarray
    n_targets_used: int

    @property
    def bias_first_order(self) -> float:
        """The one-step bias ``<h, psi_eff>_N / I_eff`` in units of ``delta``."""
        return float(self.inner_eff / self.info_eff) if self.info_eff > 0 else float("nan")


def _terms(corpus: Corpus, theta: np.ndarray, model: Model, kernel, alpha: float,
           h_fn=None, counts=None):
    """Per-target whitened pieces: ``(x_t, X_t, psi_t, N_t)`` with ``x`` the
    centred whitened mismatch, ``X`` the whitened nuisance directions and
    ``psi`` the whitened centred ``delta`` direction.

    ``counts`` overrides the response counts target by target, which is how the
    split halves are evaluated without rebuilding the corpus.
    """
    M = corpus.M
    for t, tgt in enumerate(corpus):
        n = tgt.n if counts is None else np.asarray(counts[t], dtype=np.float64)
        N = float(n.sum())
        if N <= 0:
            continue
        fit = evaluate_target(tgt, theta, model, M, kernel)
        if h_fn is not None:
            h = h_fn(tgt, fit.q)
        else:
            p_emp = (n + alpha) / (N + alpha * n.size)
            h = centre(np.log(p_emp) - np.log(np.clip(fit.q, 1e-300, None)), fit.q)
        x = whiten(centre(h, fit.q), fit.q)
        X = whiten(fit.scores[:, 1:], fit.q)
        psi = whiten(fit.scores[:, 0], fit.q)
        yield x, X, psi, N


def _solve_psd(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    if A.size == 0:
        return np.zeros(0)
    scale = max(float(np.trace(A)) / A.shape[0], 1.0)
    ridge = _RANK_TOL * scale
    for _ in range(6):
        try:
            return np.linalg.solve(A + ridge * np.eye(A.shape[0]), b)
        except np.linalg.LinAlgError:
            ridge *= 100.0
    return np.linalg.lstsq(A, b, rcond=None)[0]


def global_residual(
    corpus: Corpus,
    theta: np.ndarray,
    model: Model,
    kernel=POWER,
    alpha: float = 0.5,
    h_fn=None,
) -> GlobalResidual:
    """Project the mismatch onto the global tangent space and score it.

    Solves ``I_pp a = sum_t N_t X_t^T x_t`` once, so the residual norm is
    ``||h||_N^2 - b^T a`` and needs no second pass.  The efficient score
    direction is ``psi_t - X_t w`` with ``w = I_pp^{-1} I_pd``, and the
    residual is orthogonal to it by construction, which is why ``inner_eff``
    can be computed from ``h`` itself.
    """
    P1 = model.dim(corpus.M) - 1
    G = np.zeros((P1, P1))
    b = np.zeros(P1)
    I_pd = np.zeros(P1)
    I_dd = 0.0
    tot2 = 0.0
    inner_dh = 0.0
    per_num = per_den = 0.0
    used = 0
    for x, X, psi, N in _terms(corpus, theta, model, kernel, alpha, h_fn):
        xx = float(x @ x)
        if xx <= 0.0:
            continue
        used += 1
        G += N * (X.T @ X)
        b += N * (X.T @ x)
        I_pd += N * (X.T @ psi)
        I_dd += N * float(psi @ psi)
        tot2 += N * xx
        inner_dh += N * float(x @ psi)
        # per-target lower bound, reusing the whitened pieces
        Q, R = np.linalg.qr(X, mode="reduced") if X.shape[1] else (np.zeros((X.shape[0], 0)), None)
        if R is not None:
            d = np.abs(np.diag(R))
            Q = Q[:, d > _RANK_TOL * d.max()] if d.max() > 0 else Q[:, :0]
        perp = x - Q @ (Q.T @ x)
        per_num += N * float(perp @ perp)
        per_den += N * xx
    if used == 0:
        raise ValueError("no target contributed a non-degenerate residual")
    a = _solve_psd(G, b)
    w = _solve_psd(G, I_pd)
    perp2 = max(tot2 - float(b @ a), 0.0)
    info_eff = max(I_dd - float(I_pd @ w), 0.0)
    # <h, psi_eff>_N = <h, psi>_N - w . b, and the T component drops out.
    inner_eff = inner_dh - float(w @ b)
    align = inner_eff / np.sqrt(perp2 * info_eff) if perp2 > 0 and info_eff > 0 else float("nan")
    return GlobalResidual(
        fraction=float(np.sqrt(perp2 / tot2)),
        fraction_per_target=float(np.sqrt(per_num / per_den)),
        alignment=float(align),
        inner_eff=float(inner_eff),
        info_eff=float(info_eff),
        norm2_total=float(tot2),
        norm2_perp=float(perp2),
        coefficients=np.asarray(a, dtype=np.float64),
        n_targets_used=used,
    )


def _split_counts(n: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Split integer counts into two halves without replacement.

    Half of the responses of every target go to arm A, chosen uniformly among
    the responses, so the two arms are exchangeable and their sampling noise is
    uncorrelated to leading order, which is what removes the leading noise term
    from the cross inner product.  The split is without replacement at a fixed
    total, so the two arms are weakly negatively dependent and an O(1/N) term
    survives at N near 40, and the log transform of a smoothed proportion is
    nonlinear, so the cross inner product estimates the squared mean residual
    at half sample size rather than at the full one.  Both residues inflate the
    cross products slightly, so the debiased fraction is a corrected quantity
    rather than an unbiased one, and Table tab:alpha carries it at three
    smoothing values for that reason.
    """
    n = np.asarray(np.round(n), dtype=np.int64)
    total = int(n.sum())
    if total < 2:
        return n.astype(np.float64), np.zeros_like(n, dtype=np.float64)
    a = rng.multivariate_hypergeometric(n, total // 2)
    return a.astype(np.float64), (n - a).astype(np.float64)


def split_half_residual(
    corpus: Corpus,
    theta: np.ndarray,
    model: Model,
    kernel=POWER,
    alpha: float = 0.5,
    n_splits: int = 20,
    seed: int = 0,
) -> dict:
    """Split-half debiased global residual fraction.

    A smoothed empirical mismatch carries sampling noise whose squared norm
    inflates ``||h||^2`` and ``||h_perp||^2`` alike, and forty responses per
    context make that inflation large; a fraction near one can then be all
    noise.  Splitting each target's responses in two and taking the cross
    inner product ``<h^A_perp, h^B_perp>_N`` removes the noise term because
    the halves' errors are independent, leaving ``||h_perp||^2`` up to the
    smoothing bias, which is common to both arms and is reported through the
    ``alpha`` sensitivity instead.  The cross product of two global residuals
    is ``<x^A, x^B>_N - b_A . a_B - b_B . a_A + a_A^T I_pp a_B``.  Both cross
    products are averaged over ``n_splits`` random splits; a negative
    debiased norm is clipped to zero and counted.
    """
    rng = np.random.default_rng(seed)
    P1 = model.dim(corpus.M) - 1
    # The design pieces do not depend on the counts, so they are collected once.
    base = list(_terms(corpus, theta, model, kernel, alpha))
    if not base:
        raise ValueError("no target carries responses")
    G = np.zeros((P1, P1))
    for _, X, _, N in base:
        G += N * (X.T @ X)
    live = [t for t in corpus if t.N > 0]
    qs = [np.clip(evaluate_target(t, theta, model, corpus.M, kernel).q, 1e-300, None) for t in live]
    tot_cross, perp_cross, negatives = [], [], 0
    for _ in range(n_splits):
        bA = np.zeros(P1); bB = np.zeros(P1)
        cross = 0.0
        for (x, X, psi, N), tgt, q in zip(base, live, qs):
            nA, nB = _split_counts(tgt.n, rng)
            NA, NB = float(nA.sum()), float(nB.sum())
            if NA <= 0 or NB <= 0:
                continue
            hA = centre(np.log((nA + alpha) / (NA + alpha * nA.size)) - np.log(q), q)
            hB = centre(np.log((nB + alpha) / (NB + alpha * nB.size)) - np.log(q), q)
            xA, xB = whiten(hA, q), whiten(hB, q)
            cross += N * float(xA @ xB)
            bA += N * (X.T @ xA)
            bB += N * (X.T @ xB)
        aA, aB = _solve_psd(G, bA), _solve_psd(G, bB)
        perp = cross - float(bA @ aB) - float(bB @ aA) + float(aA @ G @ aB)
        tot_cross.append(cross)
        perp_cross.append(perp)
        if perp < 0 or cross <= 0:
            negatives += 1
    tot = float(np.mean(tot_cross))
    perp = float(np.mean(perp_cross))
    # A non-positive debiased total means the mismatch is indistinguishable
    # from sampling noise; the fraction is then undefined and reported as nan
    # with the signal-to-noise ratio beside it so that the reader can see why.
    # A negative perpendicular numerator against a positive total is the same
    # statement about the orthogonal part alone, and it is clipped to zero, so
    # a fraction reported as exactly 0.0 means "no measurable signal outside
    # the span" rather than "measured at zero"; n_negative_splits counts the
    # splits in which either quantity went non-positive.
    frac = float(np.sqrt(max(perp, 0.0) / tot)) if tot > 0 else float("nan")
    raw = sum(N * float(x @ x) for x, _, _, N in base)
    return {
        "fraction_debiased": frac,
        "norm2_total_debiased": tot,
        "norm2_perp_debiased": perp,
        "signal_share_of_norm2": float(tot / raw) if raw > 0 else float("nan"),
        "n_splits": int(n_splits),
        "n_negative_splits": int(negatives),
        "fraction_debiased_sd_over_splits": float(np.std(
            [np.sqrt(max(p, 0.0) / t) if t > 0 else np.nan for p, t in zip(perp_cross, tot_cross)])),
    }



def implied_bias(
    corpus: Corpus,
    theta: np.ndarray,
    model: Model,
    kernel=POWER,
    alpha: float = 0.5,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """The first-order bias the observed mismatch implies, with a cluster interval.

    ``delta_1 = <h, psi_eff>_N / I_eff`` is the one-step estimate of what the
    fitted decay would be if the mismatch were the whole story, so on the human
    counts it is close to the fitted ``delta`` by construction and is reported
    as a translation of the residual into the paper's units, not as evidence.
    The cluster bootstrap resamples passages and recomputes the inner product
    without refitting the nuisances, which is what keeps it cheap; ``theta``
    stays at the full-sample constrained fit.
    """
    from lcsa.kernels import d_half_from_delta

    full = global_residual(corpus, theta, model, kernel, alpha)
    C = corpus.n_clusters
    idx = corpus.cluster_index
    per_cluster_inner = np.zeros(C)
    per_cluster_info = np.zeros(C)
    per_cluster_b = np.zeros((C, full.coefficients.size))
    per_cluster_Ipd = np.zeros((C, full.coefficients.size))
    per_cluster_G = np.zeros((C, full.coefficients.size, full.coefficients.size))
    live = [t for t in corpus if t.N > 0]
    for tgt, (x, X, psi, N) in zip(live, _terms(corpus, theta, model, kernel, alpha)):
        c = int(idx[tgt.index])
        per_cluster_inner[c] += N * float(x @ psi)
        per_cluster_info[c] += N * float(psi @ psi)
        per_cluster_b[c] += N * (X.T @ x)
        per_cluster_Ipd[c] += N * (X.T @ psi)
        per_cluster_G[c] += N * (X.T @ X)
    draws = []
    for b in range(n_boot):
        pick = np.random.default_rng([int(seed), int(b)]).integers(0, C, size=C)
        G = per_cluster_G[pick].sum(axis=0)
        bb = per_cluster_b[pick].sum(axis=0)
        Ipd = per_cluster_Ipd[pick].sum(axis=0)
        w = _solve_psd(G, Ipd)
        inner = per_cluster_inner[pick].sum() - float(w @ bb)
        info = per_cluster_info[pick].sum() - float(Ipd @ w)
        draws.append(inner / info if info > 0 else np.nan)
    draws = np.asarray(draws)
    ok = np.isfinite(draws)
    d1 = full.bias_first_order
    lo, hi = (np.percentile(draws[ok], [2.5, 97.5]) if ok.sum() > 3 else (np.nan, np.nan))
    return {
        "delta_first_order": float(d1),
        "delta_first_order_lo": float(lo), "delta_first_order_hi": float(hi),
        "d_half_first_order": float(d_half_from_delta(d1, kernel=kernel)) if d1 > 0 else float("inf"),
        "inner_eff": full.inner_eff, "info_eff": full.info_eff,
        "alignment": full.alignment,
        "n_boot": int(n_boot), "n_boot_usable": int(ok.sum()),
    }


def lambda_curvature_leak(
    corpus: Corpus,
    theta: np.ndarray,
    model: Model,
    kernel=POWER,
    alpha: float = 0.5,
    a_lambda: float | None = None,
) -> dict:
    """Second-order leakage from the one curved nuisance direction (Proposition 3).

    ``beta``, ``kappa`` and ``eta`` enter ``log q`` linearly, so a tilt along
    their score directions is reproduced exactly at every order.  ``lambda``
    is the exception: ``log ptil`` is concave in ``lambda`` and a tilt of size
    ``a`` along ``s_lambda`` leaves a second-order remainder ``(a^2/2)
    c_lambda`` with ``c_lambda = -beta (u - p_delta)^2 / ptil^2``.  The bias
    it leaks into ``delta`` is ``-(a^2/2) <c_lambda, psi_eff>_N / I_eff``,
    evaluated here at ``a`` equal to the observed mismatch's projection
    coefficient on ``s_lambda`` unless ``a_lambda`` is given.  The family
    absorbs ``a s_lambda + (a^2/2) c_lambda`` exactly by moving lambda, so the
    unabsorbed residual is ``-(a^2/2) c_lambda`` and the leading minus sign
    belongs in the leak, not only in ``c_lambda`` itself.
    """
    M = corpus.M
    full = global_residual(corpus, theta, model, kernel, alpha)
    a = float(full.coefficients[0]) if a_lambda is None else float(a_lambda)
    _, lam, beta, _, _ = model.split(theta, M)
    P1 = model.dim(M) - 1
    G = np.zeros((P1, P1)); I_pd = np.zeros(P1); inner = 0.0; bc = np.zeros(P1)
    for tgt in corpus:
        if tgt.N <= 0:
            continue
        fit = evaluate_target(tgt, theta, model, M, kernel)
        c = -beta * (tgt.u - fit.p_delta) ** 2 / fit.ptil ** 2
        x = whiten(centre(c, fit.q), fit.q)
        X = whiten(fit.scores[:, 1:], fit.q)
        psi = whiten(fit.scores[:, 0], fit.q)
        G += tgt.N * (X.T @ X)
        I_pd += tgt.N * (X.T @ psi)
        inner += tgt.N * float(x @ psi)
        bc += tgt.N * (X.T @ x)
    w = _solve_psd(G, I_pd)
    inner_eff = inner - float(w @ bc)
    leak = -0.5 * a * a * inner_eff / full.info_eff if full.info_eff > 0 else float("nan")
    return {
        "a_lambda": a,
        "curvature_inner_eff": float(inner_eff),
        "delta_leak_second_order": float(leak),
        "delta_first_order": full.bias_first_order,
        "leak_over_first_order": (float(abs(leak) / abs(full.bias_first_order))
                                  if full.bias_first_order not in (0.0,) and np.isfinite(full.bias_first_order)
                                  else float("nan")),
    }


def explained_share(
    corpus: Corpus,
    theta: np.ndarray,
    model: Model,
    directions: dict[str, list[np.ndarray]],
    kernel=POWER,
    alpha: float = 0.5,
) -> dict:
    """Share of the implied bias explained by named mismatch directions.

    ``directions`` maps a name (``N-TOPIC``, ``N-ORDER``) to one direction per
    target in unwhitened units.  The residual ``h_perp`` of the observed
    mismatch is regressed, under ``<., .>_N``, on the residuals of the named
    directions with one coefficient per direction shared across targets, and
    the share is the fraction of ``<h_perp, psi_eff>_N`` that the fitted
    combination reproduces.  It is bounded above by one only when the
    directions are orthogonal to each other, so the raw value is returned
    with the Gram matrix's condition number beside it.
    """
    names = list(directions)
    M = corpus.M
    P1 = model.dim(M) - 1
    G = np.zeros((P1, P1)); I_pd = np.zeros(P1); b_h = np.zeros(P1)
    b_d = np.zeros((len(names), P1))
    live = [t for t in corpus if t.N > 0]
    rows = []
    for tgt in live:
        fit = evaluate_target(tgt, theta, model, M, kernel)
        n = tgt.n
        p_emp = (n + alpha) / (tgt.N + alpha * n.size)
        h = centre(np.log(p_emp) - np.log(np.clip(fit.q, 1e-300, None)), fit.q)
        x = whiten(h, fit.q)
        X = whiten(fit.scores[:, 1:], fit.q)
        psi = whiten(fit.scores[:, 0], fit.q)
        D = np.column_stack([whiten(centre(np.asarray(directions[nm][tgt.index]), fit.q), fit.q)
                             for nm in names])
        G += tgt.N * (X.T @ X); I_pd += tgt.N * (X.T @ psi); b_h += tgt.N * (X.T @ x)
        b_d += tgt.N * (D.T @ X)
        rows.append((tgt.N, x, X, psi, D))
    a_h = _solve_psd(G, b_h)
    w = _solve_psd(G, I_pd)
    a_d = np.stack([_solve_psd(G, b_d[k]) for k in range(len(names))])
    # Residuals of h and of each direction after the global projection.
    Kd = np.zeros((len(names), len(names))); kh = np.zeros(len(names))
    inner_h = 0.0
    for N, x, X, psi, D in rows:
        rh = x - X @ a_h
        rD = D - X @ a_d.T
        peff = psi - X @ w
        Kd += N * (rD.T @ rD); kh += N * (rD.T @ rh)
        inner_h += N * float(rh @ peff)
    coef = _solve_psd(Kd, kh)
    inner_fit = 0.0
    for N, x, X, psi, D in rows:
        rD = D - X @ a_d.T
        peff = psi - X @ w
        inner_fit += N * float((rD @ coef) @ peff)
    share = inner_fit / inner_h if inner_h != 0 else float("nan")
    cond = float(np.linalg.cond(Kd)) if Kd.size else float("nan")
    return {"directions": names, "coefficients": coef.tolist(), "share_explained": float(share),
            "gram_condition": cond, "inner_eff_h": float(inner_h),
            "inner_eff_fit": float(inner_fit)}
