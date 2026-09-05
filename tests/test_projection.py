"""Absorption geometry: the span, the decomposition, and the residual fraction."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import draw_true_delta, make_corpus

from lcsa.likelihood import NAIVE, REPAIRED, evaluate_target, nuisance_score_matrix
from lcsa.projection import (centre, corpus_residual_fraction, decompose,
                             orthogonalise_against_span, orthonormal_span,
                             residual_fraction, whiten)


@pytest.fixture(scope="module")
def data():
    return draw_true_delta(make_corpus(n_targets=40, n_clusters=8, seed=41), 0.2, seed=42)


def _q(v=8, seed=0):
    return np.random.default_rng(seed).dirichlet(np.ones(v))


def test_centre_removes_the_q_weighted_mean():
    q = _q()
    a = np.random.default_rng(1).normal(size=q.size)
    assert abs(float(q @ centre(a, q))) < 1e-14


def test_whitening_turns_the_q_inner_product_into_the_euclidean_one():
    q = _q(seed=2)
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=q.size), rng.normal(size=q.size)
    lhs = float(np.sum(q * a * b))
    assert lhs == pytest.approx(float(whiten(a, q) @ whiten(b, q)), rel=1e-12)


def test_span_is_orthonormal_and_drops_dependent_columns():
    q = _q(12, seed=4)
    rng = np.random.default_rng(5)
    B = rng.normal(size=(12, 3))
    B = np.column_stack([B, B[:, 0] + 2 * B[:, 1]])  # exactly dependent fourth column
    Q = orthonormal_span(B, q)
    assert Q.shape[1] == 3
    assert np.max(np.abs(Q.T @ Q - np.eye(3))) < 1e-10


def test_a_direction_inside_the_span_is_fully_absorbed():
    q = _q(10, seed=6)
    # The pipeline always hands in centred directions, since fit.scores are centred.
    B = centre(np.random.default_rng(7).normal(size=(10, 3)), q)
    h = B @ np.array([0.7, -1.3, 0.4])
    frac, norm = residual_fraction(h, B, q)
    assert frac < 1e-10
    assert norm > 0


def test_a_direction_outside_the_span_is_not_absorbed_at_all():
    q = _q(10, seed=8)
    B = centre(np.random.default_rng(9).normal(size=(10, 3)), q)
    # Build the vector in whitened coordinates, orthogonal both to the span and
    # to sqrt(q), which is the direction centring removes.
    Q = np.column_stack([orthonormal_span(B, q), np.sqrt(q)[:, None]])
    Q, _ = np.linalg.qr(Q)
    x = np.random.default_rng(10).normal(size=10)
    x = x - Q @ (Q.T @ x)
    h = x / np.sqrt(q)
    assert residual_fraction(h, B, q)[0] == pytest.approx(1.0, abs=1e-8)


def test_decomposition_is_additive_and_orthogonal():
    q = _q(14, seed=11)
    B = np.random.default_rng(12).normal(size=(14, 4))
    h = np.random.default_rng(13).normal(size=14)
    par, perp = decompose(h, B, q)
    assert np.max(np.abs((par + perp) - whiten(centre(h, q), q))) < 1e-12
    assert abs(float(par @ perp)) < 1e-12


def test_an_empty_span_absorbs_nothing():
    q = _q(6, seed=14)
    h = np.random.default_rng(15).normal(size=6)
    par, perp = decompose(h, np.zeros((6, 0)), q)
    assert np.max(np.abs(par)) == 0.0
    assert residual_fraction(h, np.zeros((6, 0)), q)[0] == pytest.approx(1.0)


def test_zero_mismatch_reports_nan_not_zero():
    """A responder with no mismatch has no residual, and 0 would read as absorption."""
    q = _q(7, seed=16)
    frac, norm = residual_fraction(np.zeros(7), np.random.default_rng(17).normal(size=(7, 2)), q)
    assert np.isnan(frac) and norm == 0.0


def test_richer_channel_absorbs_at_least_as_much(data):
    """Proposition 2's corollary: enlarging the nuisance set can only shrink h_perp."""
    th_n = np.array([0.0, 0.1, 0.9])
    th_r = np.concatenate([th_n, np.zeros(data.M + 1)])
    naive = corpus_residual_fraction(data, th_n, NAIVE)
    repaired = corpus_residual_fraction(data, th_r, REPAIRED)
    assert repaired.fraction <= naive.fraction + 1e-9
    assert naive.span_dim_mean == pytest.approx(2.0)
    assert repaired.span_dim_mean > naive.span_dim_mean


def test_residual_report_counts_every_usable_target(data):
    rep = corpus_residual_fraction(data, np.array([0.0, 0.1, 0.9]), NAIVE)
    assert rep.n_targets_used == len(data)
    assert 0.0 <= rep.fraction <= 1.0
    assert rep.per_target.shape == (len(data),)


def test_orthogonalised_tilts_land_outside_the_span(data):
    """This is what makes the substantive nulls un-absorbable by construction."""
    th = np.array([0.0, 0.1, 0.9])
    rng = np.random.default_rng(18)
    h_list = [rng.normal(size=t.V) for t in data]
    out = orthogonalise_against_span(data, h_list, th, NAIVE)
    for t, h in zip(data, out):
        fit = evaluate_target(t, th, NAIVE, data.M)
        B, q = nuisance_score_matrix(t, th, NAIVE, data.M)
        par, _ = decompose(h, B, q)
        assert float(np.linalg.norm(par)) < 1e-8
        assert abs(float(fit.q @ h)) < 1e-8


def test_residual_fraction_is_alpha_sensitive_but_bounded(data):
    th = np.array([0.0, 0.1, 0.9])
    fr = [corpus_residual_fraction(data, th, NAIVE, alpha=a).fraction for a in (0.1, 0.5, 1.0)]
    assert all(0.0 <= f <= 1.0 for f in fr)
    assert max(fr) - min(fr) < 0.5
