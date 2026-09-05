"""Retention kernels and the exact truncation-mixture marginalisation.

The graded random truncation family draws one ``U ~ Unif(0, 1)`` per context and
retains the ``k`` nearest words, ``k = max{d : r(d) > U}``.  The induced mask
distribution has at most ``K + 1`` atoms,

    P(k >= j) = r(j),   P(k = j) = r(j) - r(j + 1),

with the boundary conventions ``r(0) := 1`` and ``r(K + 1) := 0``.  Hence

    p_delta(w | c) = sum_{j=0}^{K} [r(j) - r(j+1)] p_ref(w | c_{last j words})

is an exact finite sum, and Abel summation gives the equivalent displacement form

    p_delta(w | c) = p_ref^{(0)}(w) + sum_{j=1}^{K} r(j) D_j(w),
    D_j(w) := p_ref^{(j)}(w) - p_ref^{(j-1)}(w).

Both forms are implemented and are asserted equal in ``tests/test_kernels.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "Kernel",
    "POWER",
    "LINEAR",
    "retention",
    "d_retention_d_delta",
    "truncation_weights",
    "marginalise",
    "marginalise_abel",
    "marginalise_and_grad",
    "d_truncation_weights",
    "get_kernel",
    "KERNELS",
    "d_half_from_delta",
    "delta_from_d_half",
    "displacements",
]


@dataclass(frozen=True)
class Kernel:
    """A one-parameter retention kernel ``r(d; delta)`` on word distances ``d >= 0``."""

    name: str
    r: Callable[[np.ndarray, float], np.ndarray]
    dr: Callable[[np.ndarray, float], np.ndarray]

    def __call__(self, d: np.ndarray, delta: float) -> np.ndarray:
        return self.r(d, delta)


def _power_r(d: np.ndarray, delta: float) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    return np.power(1.0 + d, -float(delta))


def _power_dr(d: np.ndarray, delta: float) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    return -np.log1p(d) * np.power(1.0 + d, -float(delta))


def _linear_r(d: np.ndarray, delta: float) -> np.ndarray:
    """Kuribayashi et al. (2022, App. B) linear erasure, ``r = max(1 - delta d, 0)``."""
    d = np.asarray(d, dtype=np.float64)
    return np.clip(1.0 - float(delta) * d, 0.0, 1.0)


def _linear_dr(d: np.ndarray, delta: float) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64)
    active = (1.0 - float(delta) * d > 0.0) & (1.0 - float(delta) * d < 1.0)
    return np.where(active, -d, 0.0)


POWER = Kernel("power", _power_r, _power_dr)
LINEAR = Kernel("linear", _linear_r, _linear_dr)

KERNELS = {"power": POWER, "linear": LINEAR}


def get_kernel(name: str | Kernel) -> Kernel:
    if isinstance(name, Kernel):
        return name
    try:
        return KERNELS[str(name)]
    except KeyError as exc:  # pragma: no cover - guarded by config validation
        raise ValueError(
            f"unknown kernel {name!r}; available: {sorted(KERNELS)}"
        ) from exc


def retention(d: np.ndarray, delta: float, kernel: str | Kernel = POWER) -> np.ndarray:
    """``r(d; delta)``.  At ``delta = 0`` the power kernel returns exactly 1."""
    return get_kernel(kernel).r(np.asarray(d, dtype=np.float64), float(delta))


def d_retention_d_delta(
    d: np.ndarray, delta: float, kernel: str | Kernel = POWER
) -> np.ndarray:
    """``dr/d(delta)``."""
    return get_kernel(kernel).dr(np.asarray(d, dtype=np.float64), float(delta))


def truncation_weights(
    K: int, delta: float, kernel: str | Kernel = POWER
) -> np.ndarray:
    """Atom weights ``P(k = j)`` for ``j = 0 .. K``.

    Uses ``r(0) = 1`` and ``r(K + 1) = 0``, so the weights sum to exactly 1 by
    telescoping regardless of ``delta`` or of the kernel's shape.
    """
    if K < 0:
        raise ValueError(f"K must be non-negative, got {K}")
    j = np.arange(K + 2, dtype=np.float64)
    r = retention(j, delta, kernel)
    r[0] = 1.0
    r[K + 1] = 0.0
    w = r[:-1] - r[1:]
    # Telescoping guarantees sum == 1 analytically; renormalise only against
    # accumulated floating-point error, and only if the kernel is monotone.
    if np.any(w < -1e-12):
        raise ValueError(
            f"non-monotone retention produced negative atom weight at delta={delta}"
        )
    np.clip(w, 0.0, None, out=w)
    total = w.sum()
    if total <= 0:
        raise ValueError(f"degenerate truncation weights at delta={delta}")
    return w / total


def d_truncation_weights(
    K: int, delta: float, kernel: str | Kernel = POWER
) -> np.ndarray:
    """``d P(k = j)/d delta`` for ``j = 0 .. K`` (unnormalised telescoping form)."""
    j = np.arange(K + 2, dtype=np.float64)
    dr = d_retention_d_delta(j, delta, kernel)
    dr[0] = 0.0  # r(0) == 1 identically
    dr[K + 1] = 0.0  # r(K+1) := 0 identically
    return dr[:-1] - dr[1:]


def displacements(P: np.ndarray) -> np.ndarray:
    """Incremental ablation displacements ``D_j = P[j] - P[j-1]`` for ``j >= 1``.

    Parameters
    ----------
    P : (K+1, V) array
        Row ``j`` is ``p_ref(. | c_{last j words})``.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"P must be 2-D (K+1, V), got shape {P.shape}")
    return np.diff(P, axis=0)


def marginalise(P: np.ndarray, delta: float, kernel: str | Kernel = POWER) -> np.ndarray:
    """``p_delta(. | c)`` by the atom form.  ``P`` is ``(K+1, V)``."""
    P = np.asarray(P, dtype=np.float64)
    K = P.shape[0] - 1
    w = truncation_weights(K, delta, kernel)
    return w @ P


def marginalise_abel(
    P: np.ndarray, delta: float, kernel: str | Kernel = POWER
) -> np.ndarray:
    """``p_delta`` by the Abel/displacement form; equals :func:`marginalise`."""
    P = np.asarray(P, dtype=np.float64)
    K = P.shape[0] - 1
    if K == 0:
        return P[0].copy()
    j = np.arange(1, K + 1, dtype=np.float64)
    r = retention(j, delta, kernel)
    return P[0] + r @ displacements(P)


def marginalise_and_grad(
    P: np.ndarray, delta: float, kernel: str | Kernel = POWER, D: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(p_delta, A)`` where ``A = d p_delta / d delta``.

    ``A = sum_{j=1..K} r'(j) D_j`` follows from the Abel form, in which only the
    ``r(j)`` coefficients carry ``delta``.  Passing a precomputed ``D`` avoids
    recomputing the displacements on every likelihood evaluation.
    """
    P = np.asarray(P, dtype=np.float64)
    K = P.shape[0] - 1
    if K == 0:
        return P[0].copy(), np.zeros(P.shape[1], dtype=np.float64)
    if D is None:
        D = displacements(P)
    j = np.arange(1, K + 1, dtype=np.float64)
    r = retention(j, delta, kernel)
    dr = d_retention_d_delta(j, delta, kernel)
    return P[0] + r @ D, dr @ D


def d_half_from_delta(delta: float) -> float:
    """``d_half = 2^(1/delta) - 1`` for the power kernel; ``inf`` at ``delta = 0``."""
    delta = float(delta)
    if delta <= 0:
        return float("inf")
    with np.errstate(over="ignore"):
        val = np.exp2(1.0 / delta) - 1.0
    return float(val)


def delta_from_d_half(d_half: float) -> float:
    """``delta = ln 2 / ln(1 + d_half)``; ``0`` at ``d_half = inf``."""
    d_half = float(d_half)
    if not np.isfinite(d_half):
        return 0.0
    if d_half <= 0:
        raise ValueError(f"d_half must be positive or inf, got {d_half}")
    return float(np.log(2.0) / np.log1p(d_half))
