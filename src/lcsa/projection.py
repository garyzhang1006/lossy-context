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
    """``h = log(smoothed empirical) - log p_ref^{(0)}``, centred under ``q``.

    Jeffreys smoothing with ``alpha = 0.5`` keeps ``h`` finite when a candidate
    drew zero responses, which is the common case on a 120-type candidate set
    with 40 responses.  The sensitivity of the reported fraction to
    ``alpha in {0.1, 0.5, 1.0}`` is what ``experiments/e3`` records.
    """
    n = tgt.n
    p_emp = (n + alpha) / (n.sum() + alpha * n.size)
    h = np.log(p_emp) - np.log(tgt.P[0])
    return centre(h, q)


def corpus_residual_fraction(
    corpus: Corpus,
    theta: np.ndarray,
    model: Model,
    kernel=POWER,
    alpha: float = 0.5,
    h_fn=None,
) -> ResidualReport:
    """Residual fraction of the observed mismatch against the fitted nuisance span.

    The headline number is the response-count-weighted root of the ratio of
    summed squared norms, which is the quantity that enters the bias expression
    of Proposition 2; the unweighted mean over targets is reported beside it so
    that a handful of high-count passages cannot drive the fraction alone.
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
