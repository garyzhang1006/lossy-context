"""The two estimators, their scores, and their information."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import draw_true_delta, make_corpus

from lcsa.likelihood import (NAIVE, REPAIRED, evaluate_target, get_model,
                             information, loglik, loglik_and_grad,
                             nuisance_score_matrix, observed_scores)


@pytest.fixture(scope="module")
def data():
    c = make_corpus(n_targets=40, n_clusters=8, seed=21)
    return draw_true_delta(c, 0.3, seed=22)


def theta_for(model, M, delta=0.25, seed=0):
    rng = np.random.default_rng(seed)
    th = np.zeros(model.dim(M))
    th[:3] = [delta, 0.12, 0.85]
    if th.size > 3:
        th[3:] = rng.normal(scale=0.3, size=th.size - 3)
    return th


@pytest.mark.parametrize("model", [NAIVE, REPAIRED])
def test_fitted_distribution_is_normalised(data, model):
    for t in data:
        fit = evaluate_target(t, theta_for(model, data.M), model, data.M)
        assert fit.q.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(fit.q > 0)
        assert fit.log_q == pytest.approx(np.log(fit.q), abs=1e-12)


@pytest.mark.parametrize("model", [NAIVE, REPAIRED])
def test_scores_are_centred_under_q(data, model):
    """Every score direction must have mean zero under the fitted distribution."""
    for t in data:
        fit = evaluate_target(t, theta_for(model, data.M), model, data.M)
        assert np.max(np.abs(fit.q @ fit.scores)) < 1e-12


@pytest.mark.parametrize("model", [NAIVE, REPAIRED])
@pytest.mark.parametrize("delta", [0.0, 0.25, 1.1])
def test_gradient_matches_finite_differences(data, model, delta):
    th = theta_for(model, data.M, delta=delta, seed=3)
    _, g = loglik_and_grad(data, th, model)
    eps = 1e-6
    fd = np.zeros_like(g)
    for i in range(g.size):
        a, b = th.copy(), th.copy()
        # delta is bounded below at zero, so difference one-sided when sitting on it.
        if i == 0 and delta == 0.0:
            fd[i] = (loglik(data, b + np.eye(g.size)[i] * eps, model) - loglik(data, th, model)) / eps
            continue
        a[i] += eps
        b[i] -= eps
        fd[i] = (loglik(data, a, model) - loglik(data, b, model)) / (2 * eps)
    scale = max(1.0, np.max(np.abs(g)))
    assert np.max(np.abs(g - fd)) / scale < 1e-5


def test_observed_scores_sum_to_the_gradient(data):
    th = theta_for(REPAIRED, data.M, seed=5)
    _, g = loglik_and_grad(data, th, REPAIRED)
    U = observed_scores(data, th, REPAIRED)
    assert U.shape == (len(data), REPAIRED.dim(data.M))
    assert np.max(np.abs(U.sum(axis=0) - g)) < 1e-9


def test_delta_score_is_the_covariance_of_the_paper(data):
    """Equation 5: the score is a within-context covariance of counts with A/ptilde."""
    th = theta_for(NAIVE, data.M, delta=0.2)
    total = 0.0
    for t in data:
        fit = evaluate_target(t, th, NAIVE, data.M)
        direction = fit.A / fit.ptil
        centred = direction - float(fit.q @ direction)
        total += th[2] * (1 - th[1]) * float(t.n @ centred)
    assert loglik_and_grad(data, th, NAIVE)[1][0] == pytest.approx(total, rel=1e-9)


def test_information_is_positive_semidefinite_and_symmetric(data):
    I = information(data, theta_for(REPAIRED, data.M, seed=6), REPAIRED)
    assert np.max(np.abs(I - I.T)) < 1e-12
    assert np.min(np.linalg.eigvalsh(I)) > -1e-9


def test_nuisance_span_has_the_dimensions_the_paper_claims(data):
    """Two directions for the naive estimator and seven for the repaired one."""
    t = data.target(0)
    Bn, _ = nuisance_score_matrix(t, theta_for(NAIVE, data.M), NAIVE, data.M)
    Br, _ = nuisance_score_matrix(t, theta_for(REPAIRED, data.M), REPAIRED, data.M)
    assert Bn.shape[1] == 2
    assert Br.shape[1] == 3 + data.M  # lam, beta, four kappa, eta, minus delta
    assert REPAIRED.dim(data.M) == 8


def test_repaired_reduces_to_naive_when_its_extra_channels_are_off(data):
    """The naive estimator is the repaired one at kappa = 0, eta = 0, not a different model."""
    base = np.array([0.3, 0.15, 0.8])
    full = np.concatenate([base, np.zeros(data.M + 1)])
    assert loglik(data, full, REPAIRED) == pytest.approx(loglik(data, base, NAIVE), rel=1e-12)


def test_beta_one_lambda_zero_is_the_bare_reference(data):
    """With no temperature and no unigram floor the estimator is exactly p_delta."""
    t = data.target(0)
    fit = evaluate_target(t, np.array([0.4, 1e-6, 1.0]), NAIVE, data.M)
    assert np.max(np.abs(fit.q - fit.p_delta)) < 1e-5


def test_zero_count_targets_contribute_nothing(data):
    counts = [t.n.copy() for t in data]
    counts[0] = np.zeros_like(counts[0])
    stripped = data.with_counts(counts)
    th = theta_for(NAIVE, data.M)
    drop = float(data.target(0).n @ evaluate_target(data.target(0), th, NAIVE, data.M).log_q)
    assert loglik(stripped, th, NAIVE) == pytest.approx(loglik(data, th, NAIVE) - drop, rel=1e-12)


def test_wrong_length_theta_is_refused(data):
    with pytest.raises(ValueError, match="expects theta"):
        loglik(data, np.array([0.2, 0.1, 1.0]), REPAIRED)


def test_get_model_rejects_unknown_names():
    assert get_model("naive") is NAIVE
    with pytest.raises(ValueError, match="unknown model"):
        get_model("repaired-v2")


def test_likelihood_prefers_the_truth_over_a_wrong_delta():
    """A sanity check that the likelihood surface points the right way at all."""
    c = draw_true_delta(make_corpus(n_targets=120, n_clusters=10, seed=31), 0.6,
                        seed=32, n_per_target=200)
    at_truth = loglik(c, np.array([0.6, 0.1, 0.9]), NAIVE)
    at_zero = loglik(c, np.array([0.0, 0.1, 0.9]), NAIVE)
    assert at_truth > at_zero
