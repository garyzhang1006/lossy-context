"""Reliability arithmetic: divergences that survive finite response counts.

Two phenotypes are measured before any fit, because both bound what the design
can possibly show.

*Cloze reliability.*  The Jensen-Shannon divergence between two halves of the
human responses, against the divergence between the humans and the reference.
A plug-in JS is biased upward at finite counts even when both arguments come
from the same distribution, and at Provo's roughly 40 responses per target the
absolute divergence is simply not identified to better than the spread between
the available estimators.  Three are computed and all three are reported: the
raw plug-in, the closed form ``(S_eff - 1)/(4m)`` usually quoted, and a
parametric bootstrap correction.  The closed form is optimistic because it
assumes every type carries appreciable probability, and the bootstrap is
optimistic in the other direction because resampling from the observed support
cannot reproduce types the sample missed.

Nothing in the design depends on resolving that.  The substantive nulls are
calibrated by :func:`js_statistic`, applied *identically* to the human counts and
to simulated null counts at the same ``N_t`` on the same contexts, so whatever
bias the statistic carries is common to both sides and cancels in the match.
An absolute JS is reported as a descriptive number and never as the basis of a
comparison.

*Over-dispersion.*  Splitting one target's responses in half cannot reveal
over-dispersion at that target: both halves are drawn from the same latent rate,
so the split-half divergence is identical to its multinomial value however
strongly the target's responses cluster.  What the naive likelihood ratio
actually mistakes is dependence *between* responses in the same passage, so the
quantity to match is the between-cluster design effect of the efficient score,
computed by :func:`lcsa.inference.design_effect` and used to calibrate the
over-dispersed floor N0-PRIME in :mod:`lcsa.readers`.  The split-half numbers
below stay in the design as the cloze reliability phenotype; they are not an
over-dispersion measurement and are not used as one.

*Gaze reliability.*  The Spearman-Brown corrected split-half correlation of gaze
duration, which caps the reading-time experiment's effect size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "js_divergence",
    "js_statistic",
    "split_counts",
    "js_reliability",
    "analytic_js_floor",
    "effective_types",
    "spearman_brown",
    "split_half_reliability",
    "ReliabilityReport",
]

_EPS = 1e-300


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats, between 0 and ln 2."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return float("nan")
    p, q = p / sp, q / sq
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / np.clip(b[mask], _EPS, None))).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def effective_types(counts: np.ndarray) -> float:
    """Inverse Simpson index: the number of types the responses effectively spread over."""
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum()
    if n <= 0:
        return 0.0
    p = c / n
    return float(1.0 / np.clip((p ** 2).sum(), 1e-12, None))


def analytic_js_floor(counts: np.ndarray) -> float:
    """``(S_eff - 1)/(4m)`` with ``m`` the half count: the quoted closed form.

    Reported for comparison only.  :func:`js_reliability` debiases by simulation
    because this expression is optimistic whenever the candidate set has a long
    tail of rare types, which is the normal case here.
    """
    c = np.asarray(counts, dtype=np.float64)
    n = float(c.sum())
    if n < 2:
        return float("nan")
    return float((effective_types(c) - 1.0) / (4.0 * (n / 2.0)))


def split_counts(counts: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Split a count vector into two halves by multivariate hypergeometric thinning.

    Thinning respects the multinomial structure exactly, unlike splitting the
    participant list, which is unavailable once responses are aggregated.
    """
    c = np.asarray(counts, dtype=np.int64)
    n = int(c.sum())
    if n < 2:
        z = np.zeros(c.size, dtype=np.float64)
        return z, z.copy()
    a = np.zeros(c.size, dtype=np.int64)
    remaining_total = n
    remaining_draw = n // 2
    for i, ci in enumerate(c):
        if remaining_draw <= 0:
            break
        ci = int(ci)
        if ci <= 0:
            continue
        take = int(rng.hypergeometric(ci, max(remaining_total - ci, 0), remaining_draw))
        a[i] = take
        remaining_total -= ci
        remaining_draw -= take
    return a.astype(np.float64), (c - a).astype(np.float64)


def _plugin(counts: np.ndarray) -> np.ndarray:
    """Plug-in rate for the parametric bootstrap.

    Deliberately unsmoothed.  Adding 0.5 per type to a 40-response vector on a
    120-type candidate set adds 60 pseudo-counts, which flattens the resampling
    distribution and makes the estimated bias larger than the statistic it is
    correcting.  The plug-in is the standard choice and is the one that
    reproduces the observed statistic's sampling behaviour.
    """
    c = np.asarray(counts, dtype=np.float64)
    total = c.sum()
    if total <= 0:
        raise ValueError("cannot bootstrap from an all-zero count vector")
    return c / total


def _mean_split_js(counts: np.ndarray, rng: np.random.Generator, n_splits: int) -> float:
    vals = []
    for _ in range(n_splits):
        a, b = split_counts(counts, rng)
        if a.sum() > 0 and b.sum() > 0:
            vals.append(js_divergence(a, b))
    return float(np.mean(vals)) if vals else float("nan")


def js_statistic(count_list: list[np.ndarray], reference_list: list[np.ndarray]) -> float:
    """Mean plug-in JS between counts and reference, over targets with responses.

    This is the calibration statistic.  It is deliberately the plain plug-in:
    the substantive nulls are matched to the human value through this same
    function evaluated on simulated counts of the same size, so a bias shared by
    both sides is irrelevant and a cleverer estimator would only add variance.
    """
    vals = []
    for c, r in zip(count_list, reference_list):
        c = np.asarray(c, dtype=np.float64)
        if c.sum() <= 0:
            continue
        v = js_divergence(c, np.asarray(r, dtype=np.float64))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        raise ValueError("no target had responses, so the JS statistic is undefined")
    return float(np.mean(vals))


def js_reliability(
    count_list: list[np.ndarray],
    reference_list: list[np.ndarray] | None = None,
    n_splits: int = 20,
    n_sim: int = 20,
    seed: int = 0,
) -> dict:
    """Split-half JS, human-versus-reference JS, and both debiased by simulation.

    The bias of each plug-in statistic is estimated by resampling counts from the
    smoothed empirical distribution at the observed count and recomputing the
    same statistic, so the two corrections differ, as they must: a half-versus-
    half comparison carries noise from both arguments and a half-versus-reference
    comparison from only one.
    """
    rng = np.random.default_rng(seed)
    sh, sh_bias, refs, ref_bias, floors = [], [], [], [], []
    for i, c in enumerate(count_list):
        c = np.rint(np.asarray(c, dtype=np.float64))
        n = int(c.sum())
        if n < 4:
            continue
        p_hat = _plugin(c)
        obs = _mean_split_js(c, rng, n_splits)
        if not np.isfinite(obs):
            continue
        sim = [
            _mean_split_js(rng.multinomial(n, p_hat).astype(np.float64), rng, max(n_splits // 4, 3))
            for _ in range(n_sim)
        ]
        sh.append(obs)
        sh_bias.append(float(np.nanmean(sim)))
        floors.append(analytic_js_floor(c))
        if reference_list is not None:
            ref = np.asarray(reference_list[i], dtype=np.float64)
            refs.append(js_divergence(c, ref))
            ref_bias.append(
                float(
                    np.mean(
                        [js_divergence(rng.multinomial(n, p_hat).astype(np.float64), p_hat)
                         for _ in range(n_sim)]
                    )
                )
            )
    out = {
        "js_split_half": float(np.mean(sh)) if sh else float("nan"),
        "js_split_half_bias": float(np.mean(sh_bias)) if sh_bias else float("nan"),
        "js_split_half_debiased": (
            float(np.mean(sh) - np.mean(sh_bias)) if sh else float("nan")
        ),
        "js_analytic_floor": float(np.nanmean(floors)) if floors else float("nan"),
        "n_targets": len(sh),
    }
    if refs:
        out["js_vs_reference"] = float(np.mean(refs))
        out["js_vs_reference_bias"] = float(np.mean(ref_bias))
        out["js_vs_reference_debiased"] = float(np.mean(refs) - np.mean(ref_bias))
    return out


def spearman_brown(r: float, k: float = 2.0) -> float:
    """Spearman-Brown prophecy: reliability of a ``k``-times longer test."""
    if not np.isfinite(r):
        return float("nan")
    den = 1.0 + (k - 1.0) * r
    return float(k * r / den) if abs(den) > 1e-12 else float("nan")


def split_half_reliability(
    values: np.ndarray, groups: np.ndarray, seed: int = 0, n_splits: int = 50,
    participants: np.ndarray | None = None,
) -> float:
    """Spearman-Brown corrected split-half correlation of per-item means.

    With ``participants`` the halves are halves of the participant set, which
    is the registered form for gaze (84 readers into two groups of 42); without
    it each observation is assigned to a half independently.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    if participants is not None:
        pu, pinv = np.unique(np.asarray(participants), return_inverse=True)
        if pu.size < 2:
            return float("nan")
    rs = []
    for _ in range(n_splits):
        if participants is not None:
            half = rng.permutation(pu.size) < pu.size // 2
            pick = half[pinv]
        else:
            pick = rng.random(values.size) < 0.5
        a = np.bincount(inv[pick], weights=values[pick], minlength=uniq.size)
        na = np.bincount(inv[pick], minlength=uniq.size)
        b = np.bincount(inv[~pick], weights=values[~pick], minlength=uniq.size)
        nb = np.bincount(inv[~pick], minlength=uniq.size)
        ok = (na > 0) & (nb > 0)
        if ok.sum() < 3:
            continue
        r = np.corrcoef(a[ok] / na[ok], b[ok] / nb[ok])[0, 1]
        if np.isfinite(r):
            rs.append(r)
    if not rs:
        return float("nan")
    return spearman_brown(float(np.mean(rs)))


@dataclass
class ReliabilityReport:
    js_split_half: float
    js_split_half_debiased: float
    js_analytic_floor: float
    js_vs_reference: float
    js_vs_reference_debiased: float
    gaze_split_half: float
    design_effect: float
    n_targets: int
