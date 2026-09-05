"""Identifiability audit of context-decay parameters fitted through LM likelihoods.

Package layout
--------------
kernels      retention kernels and exact truncation-mixture marginalisation
corpusdata   ragged in-memory container for the nested ablation cache
likelihood   naive and repaired estimators, log-likelihood, analytic scores
projection   nuisance-score span B_t, weighted thin QR, residual fractions
fitting      constrained/unconstrained MLE and profile-likelihood regions
inference    cluster-robust efficient-score test, CR1/CR3, bootstrap, TOST
readers      the ten fitted objects plus the recovery ladder
"""

__version__ = "0.1.0"

from lcsa.kernels import (  # noqa: F401
    POWER,
    LINEAR,
    d_half_from_delta,
    delta_from_d_half,
    marginalise,
    retention,
    truncation_weights,
)

__all__ = [
    "__version__",
    "POWER",
    "LINEAR",
    "retention",
    "truncation_weights",
    "marginalise",
    "d_half_from_delta",
    "delta_from_d_half",
]
