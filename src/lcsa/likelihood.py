"""The two estimators, their log-likelihood, and their scores in closed form.

Naive:     q^N(w|c) proportional to [(1-lam) p_delta(w|c) + lam u(w)]^beta
Repaired:  q^R(w|c) proportional to [ ... ]^beta exp(kappa'f(w) + eta g(w,c))

Parameter vector, in this order:

    theta = [delta, lam, beta]                      (naive,    dim 3)
    theta = [delta, lam, beta, kappa_1..kappa_M, eta]  (repaired, dim 4 + M)

with ``phi`` denoting every coordinate except ``delta``.

Every derivative below is analytic.  Writing ``logqtil = beta log ptil +
kappa'f + eta g`` and ``q = softmax(logqtil)``, the gradient of the multinomial
log-likelihood is

    dL/dtheta = sum_t sum_w n_tw [ d logqtil_tw/dtheta - E_q(d logqtil/dtheta) ],

so every score is a *centred* direction under ``q_t`` and the delta coordinate
reduces to Eq. 5 of the paper,

    dL/ddelta = beta (1 - lam) sum_t sum_w n_tw [ A_t/ptil_t - E_q(A_t/ptil_t) ].

``tests/test_likelihood.py`` checks the full gradient against central finite
differences at several random points for both estimators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from lcsa.corpusdata import PROB_FLOOR, Corpus, Target
from lcsa.kernels import POWER, Kernel, d_retention_d_delta, get_kernel, retention

__all__ = [
    "Model",
    "NAIVE",
    "REPAIRED",
    "TargetFit",
    "evaluate_target",
    "loglik",
    "loglik_and_grad",
    "nuisance_score_matrix",
    "delta_score_vector",
    "information",
    "observed_scores",
]


@dataclass(frozen=True)
class Model:
    """Which nuisance channels the estimator carries."""

    name: str
    lexical: bool  # kappa'f(w)
    prior_mention: bool  # eta g(w,c)

    def dim(self, M: int) -> int:
        return 3 + (M if self.lexical else 0) + (1 if self.prior_mention else 0)

    def param_names(self, M: int, feature_names: list[str] | None = None) -> list[str]:
        names = ["delta", "lam", "beta"]
        if self.lexical:
            fn = feature_names or [f"f{i}" for i in range(M)]
            names += [f"kappa[{n}]" for n in fn]
        if self.prior_mention:
            names += ["eta"]
        return names

    def split(self, theta: np.ndarray, M: int) -> tuple[float, float, float, np.ndarray, float]:
        theta = np.asarray(theta, dtype=np.float64)
        want = self.dim(M)
        if theta.shape != (want,):
            raise ValueError(
                f"model {self.name!r} with M={M} expects theta of shape ({want},), "
                f"got {theta.shape}"
            )
        delta, lam, beta = float(theta[0]), float(theta[1]), float(theta[2])
        i = 3
        if self.lexical:
            kappa = theta[i : i + M]
            i += M
        else:
            kappa = np.zeros(0, dtype=np.float64)
        eta = float(theta[i]) if self.prior_mention else 0.0
        return delta, lam, beta, kappa, eta

    def start(self, M: int) -> np.ndarray:
        """Neutral starting point: no decay, no unigram floor, temperature 1."""
        x = np.zeros(self.dim(M), dtype=np.float64)
        x[0] = 0.0
        x[1] = 0.05
        x[2] = 1.0
        return x

    def bounds(self, M: int, delta_max: float = 5.0) -> list[tuple[float, float]]:
        b = [(0.0, delta_max), (1e-6, 1.0 - 1e-6), (0.05, 10.0)]
        if self.lexical:
            b += [(-10.0, 10.0)] * M
        if self.prior_mention:
            b += [(-10.0, 10.0)]
        return b


NAIVE = Model("naive", lexical=False, prior_mention=False)
REPAIRED = Model("repaired", lexical=True, prior_mention=True)

MODELS = {"naive": NAIVE, "repaired": REPAIRED}


def get_model(name: str | Model) -> Model:
    if isinstance(name, Model):
        return name
    try:
        return MODELS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}") from exc


@dataclass
class TargetFit:
    """Everything one target contributes at a parameter value."""

    q: np.ndarray  # (V,) fitted response distribution
    log_q: np.ndarray  # (V,)
    ptil: np.ndarray  # (V,) mixed reference
    p_delta: np.ndarray  # (V,)
    A: np.ndarray  # (V,) d p_delta / d delta
    scores: np.ndarray  # (V, P) centred score directions, delta first
    N: float


def _mixture(tgt: Target, delta: float, kernel: Kernel) -> tuple[np.ndarray, np.ndarray]:
    """``(p_delta, A)`` via the Abel form using the cached displacements."""
    K = tgt.K
    if K == 0:
        V = tgt.V
        return tgt.P[0], np.zeros(V, dtype=np.float64)
    j = np.arange(1, K + 1, dtype=np.float64)
    r = retention(j, delta, kernel)
    dr = d_retention_d_delta(j, delta, kernel)
    return tgt.P[0] + r @ tgt.D, dr @ tgt.D


def evaluate_target(
    tgt: Target,
    theta: np.ndarray,
    model: Model,
    M: int,
    kernel: str | Kernel = POWER,
) -> TargetFit:
    """Fitted distribution and every centred score direction at ``theta``."""
    kern = get_kernel(kernel)
    delta, lam, beta, kappa, eta = model.split(theta, M)

    p_delta, A = _mixture(tgt, delta, kern)
    p_delta = np.clip(p_delta, PROB_FLOOR, None)
    ptil = (1.0 - lam) * p_delta + lam * tgt.u
    ptil = np.clip(ptil, PROB_FLOOR, None)
    log_ptil = np.log(ptil)

    log_qtil = beta * log_ptil
    if model.lexical:
        log_qtil = log_qtil + tgt.f @ kappa
    if model.prior_mention:
        log_qtil = log_qtil + eta * tgt.g
    log_q = log_qtil - logsumexp(log_qtil)
    q = np.exp(log_q)

    # Raw (uncentred) directions d logqtil / d theta.
    raw = [
        beta * (1.0 - lam) * (A / ptil),  # delta
        beta * (tgt.u - p_delta) / ptil,  # lam
        log_ptil,  # beta
    ]
    if model.lexical:
        raw.extend(tgt.f.T)
    if model.prior_mention:
        raw.append(tgt.g)
    S = np.stack(raw, axis=1)  # (V, P)
    S = S - (q @ S)  # centre under q_t
    return TargetFit(
        q=q, log_q=log_q, ptil=ptil, p_delta=p_delta, A=A, scores=S, N=tgt.N
    )


def loglik_and_grad(
    corpus: Corpus,
    theta: np.ndarray,
    model: Model,
    kernel: str | Kernel = POWER,
) -> tuple[float, np.ndarray]:
    """Total log-likelihood and its exact gradient."""
    theta = np.asarray(theta, dtype=np.float64)
    M = corpus.M
    P = model.dim(M)
    total = 0.0
    grad = np.zeros(P, dtype=np.float64)
    for tgt in corpus:
        if tgt.N == 0:
            continue
        fit = evaluate_target(tgt, theta, model, M, kernel)
        total += float(tgt.n @ fit.log_q)
        grad += tgt.n @ fit.scores
    if not np.isfinite(total):
        raise FloatingPointError(
            f"non-finite log-likelihood at theta={np.asarray(theta).tolist()}"
        )
    return total, grad


def loglik(
    corpus: Corpus, theta: np.ndarray, model: Model, kernel: str | Kernel = POWER
) -> float:
    return loglik_and_grad(corpus, theta, model, kernel)[0]


def observed_scores(
    corpus: Corpus,
    theta: np.ndarray,
    model: Model,
    kernel: str | Kernel = POWER,
) -> np.ndarray:
    """Per-target observed score contributions ``U_t = sum_w n_tw s(w)``, shape (T, P)."""
    M = corpus.M
    U = np.zeros((len(corpus), model.dim(M)), dtype=np.float64)
    for t, tgt in enumerate(corpus):
        if tgt.N == 0:
            continue
        fit = evaluate_target(tgt, theta, model, M, kernel)
        U[t] = tgt.n @ fit.scores
    return U


def information(
    corpus: Corpus,
    theta: np.ndarray,
    model: Model,
    kernel: str | Kernel = POWER,
) -> np.ndarray:
    """Expected information ``I(a,b) = sum_t N_t E_q[s_a s_b]``, shape (P, P).

    This is the multinomial Fisher information at ``theta`` with the observed
    response counts as the trial counts, which is what the score test's
    projection needs; it is positive semi-definite by construction.
    """
    M = corpus.M
    P = model.dim(M)
    I = np.zeros((P, P), dtype=np.float64)
    for tgt in corpus:
        if tgt.N == 0:
            continue
        fit = evaluate_target(tgt, theta, model, M, kernel)
        I += tgt.N * (fit.scores.T * fit.q) @ fit.scores
    return 0.5 * (I + I.T)


def information_by_cluster(
    corpus: Corpus,
    theta: np.ndarray,
    model: Model,
    kernel: str | Kernel = POWER,
) -> np.ndarray:
    """Per-cluster information blocks, shape (C, P, P).  Used by the CR3 leverage."""
    M = corpus.M
    P = model.dim(M)
    out = np.zeros((corpus.n_clusters, P, P), dtype=np.float64)
    for tgt in corpus:
        if tgt.N == 0:
            continue
        fit = evaluate_target(tgt, theta, model, M, kernel)
        blk = tgt.N * (fit.scores.T * fit.q) @ fit.scores
        out[tgt.cluster] += 0.5 * (blk + blk.T)
    return out


def nuisance_score_matrix(
    tgt: Target,
    theta: np.ndarray,
    model: Model,
    M: int,
    kernel: str | Kernel = POWER,
) -> tuple[np.ndarray, np.ndarray]:
    """``(B_raw, q)`` where ``B_raw`` is (V, P-1), the centred nuisance directions."""
    fit = evaluate_target(tgt, theta, model, M, kernel)
    return fit.scores[:, 1:], fit.q


def delta_score_vector(
    tgt: Target,
    theta: np.ndarray,
    model: Model,
    M: int,
    kernel: str | Kernel = POWER,
) -> np.ndarray:
    """The centred ``delta`` direction ``A_t/ptil_t - E_q(A_t/ptil_t)`` times ``beta(1-lam)``."""
    return evaluate_target(tgt, theta, model, M, kernel).scores[:, 0]
