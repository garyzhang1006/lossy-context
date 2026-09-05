"""Cluster-robust score test, the naive likelihood ratio, and the bootstrap."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import draw, draw_true_delta, make_corpus

from lcsa.fitting import fit_constrained
from lcsa.inference import (cluster_bootstrap, lr_test, rejection_rate, score_test,
                            tost)
from lcsa.likelihood import NAIVE, evaluate_target, observed_scores
from lcsa.readers import reader_n0_prime


@pytest.fixture(scope="module")
def base():
    return make_corpus(n_targets=60, n_clusters=12, seed=51)


@pytest.fixture(scope="module")
def floor(base):
    """The plain floor: responses drawn i.i.d. from the fitted family at delta = 0."""
    return draw_true_delta(base, 0.0, seed=52)


def test_numerator_is_unchanged_by_the_nuisance_projection(floor):
    """Nuisance score sums vanish at the constrained fit, so only the denominator moves."""
    res = score_test(floor, NAIVE)
    U = observed_scores(floor, res.null_fit.theta, NAIVE)
    assert np.max(np.abs(U[:, 1:].sum(axis=0))) < 1e-5
    assert res.numerator == pytest.approx(float(U[:, 0].sum()), rel=1e-6)


def test_unprojected_statistic_is_the_conservative_one(floor):
    """The projection shrinks the denominator, so the raw form cannot reject more."""
    res = score_test(floor, NAIVE)
    assert res.T_raw <= res.T_cr1 + 1e-8


def test_headline_takes_the_more_conservative_of_cr1_and_cr3(floor):
    res = score_test(floor, NAIVE)
    assert res.T == min(res.T_cr1, res.T_cr3)
    assert res.p == max(res.p_cr1, res.p_cr3)
    assert 0.0 <= res.p <= 1.0


def test_design_effect_is_about_one_on_independent_responses(floor):
    """With no over-dispersion the cluster-robust variance matches the model-based one."""
    res = score_test(floor, NAIVE)
    assert 0.2 < res.design_effect < 3.0


def test_over_dispersion_inflates_the_design_effect(floor):
    """Passage-level tilts are exactly what the cluster-robust denominator exists for."""
    theta0 = np.array([0.0, 0.1, 0.9])
    plain = np.median([score_test(reader_n0_prime(floor, theta0, NAIVE, sigma_passage=0.0,
                                                 seed=60 + i), NAIVE).design_effect
                       for i in range(4)])
    tilted = np.median([score_test(reader_n0_prime(floor, theta0, NAIVE, sigma_passage=0.9,
                                                   seed=60 + i), NAIVE).design_effect
                        for i in range(4)])
    assert plain < 2.0 < tilted


def test_floor_rejects_rarely(base):
    """Prediction 1 in miniature: the plain floor must not reject often."""
    p = [score_test(draw_true_delta(base, 0.0, seed=100 + i), NAIVE).p for i in range(12)]
    rate, lo, hi = rejection_rate(p)
    assert rate <= 0.25
    assert lo <= rate <= hi


def test_score_test_detects_real_decay():
    """A corpus generated at a half-distance of about three words must reject."""
    c = draw_true_delta(make_corpus(n_targets=120, n_clusters=12, seed=71), 0.5,
                        seed=72, n_per_target=120)
    assert score_test(c, NAIVE).p < 0.05


def test_one_cluster_is_refused(base):
    single = draw_true_delta(base, 0.0, seed=80).subset_clusters([0])
    with pytest.raises(ValueError, match="at least 2 clusters"):
        score_test(single, NAIVE)


def test_likelihood_ratio_uses_the_boundary_mixture(floor):
    r = lr_test(floor, NAIVE)
    assert r.LR >= 0.0
    assert 0.0 <= r.p <= 1.0
    if r.delta_hat == 0.0:
        assert r.p == 1.0


def test_likelihood_ratio_rejects_harder_than_the_robust_test_under_over_dispersion(floor):
    """Prediction 2 in miniature: this gap is the whole reason for the repair."""
    theta0 = np.array([0.0, 0.1, 0.9])
    lr_hits = robust_hits = 0
    for i in range(6):
        c = reader_n0_prime(floor, theta0, NAIVE, sigma_passage=1.2, seed=90 + i)
        lr_hits += lr_test(c, NAIVE).rejects()
        robust_hits += score_test(c, NAIVE).rejects()
    assert lr_hits > robust_hits


def test_bootstrap_returns_one_record_per_replicate(floor):
    recs = cluster_bootstrap(floor, lambda c: {"delta": fit_constrained(c, NAIVE, n_starts=1).delta},
                             n_boot=6, seed=1)
    assert len(recs) == 6
    assert [r["replicate"] for r in recs] == list(range(6))
    assert all(r["error"] is None for r in recs)


def test_bootstrap_records_failures_instead_of_dropping_them(floor):
    def broken(c):
        raise RuntimeError("no convergence")

    recs = cluster_bootstrap(floor, broken, n_boot=4, seed=2)
    assert len(recs) == 4
    assert all("no convergence" in r["error"] for r in recs)


def test_bootstrap_resamples_clusters_with_replacement(floor):
    """Replicates must differ from each other, or the interval they produce is fake."""
    recs = cluster_bootstrap(
        floor, lambda c: {"C": c.n_clusters,
                          "signature": float(sum(t.P[0, 0] for t in c))},
        n_boot=8, seed=3)
    assert all(r["C"] <= floor.n_clusters for r in recs)
    assert len({round(r["signature"], 9) for r in recs}) > 1


def test_tost_declares_equivalence_only_when_the_difference_is_inside_the_margin():
    tight = np.random.default_rng(4).normal(0.0, 0.03, 200)
    wide = np.random.default_rng(5).normal(0.7, 0.10, 200)
    assert tost(tight, margin=0.25).equivalent
    assert not tost(wide, margin=0.25).equivalent


def test_tost_refuses_to_speak_on_too_few_usable_replicates():
    r = tost([0.01, float("nan"), float("inf")], margin=0.25)
    assert not r.equivalent and r.n_used == 1


def test_rejection_rate_interval_brackets_the_rate():
    rate, lo, hi = rejection_rate([0.01] * 3 + [0.9] * 7)
    assert rate == pytest.approx(0.3)
    assert 0.0 <= lo < rate < hi <= 1.0
    assert np.isnan(rejection_rate([])[0])
