"""Absorption geometry: the span, the decomposition, and the residual fraction."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import draw_true_delta, make_corpus

from lcsa.likelihood import NAIVE, REPAIRED, evaluate_target, nuisance_score_matrix
from lcsa.projection import (centre, corpus_residual_fraction, decompose,
                             explained_share, global_residual, implied_bias,
                             lambda_curvature_leak, orthogonalise_against_span,
                             orthonormal_span, residual_fraction,
                             split_half_residual, whiten)


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


# -- the global tangent space -------------------------------------------------


def _h_from_nuisance_shift(corpus, theta, model, shift):
    """A mismatch that a single shared nuisance shift reproduces to first order."""
    out = []
    for t in corpus:
        fit = evaluate_target(t, theta, model, corpus.M)
        out.append(centre(fit.scores[:, 1:] @ shift, fit.q))
    return out


def test_per_target_fraction_lower_bounds_the_global_one(data):
    th = np.array([0.0, 0.1, 0.9])
    g = global_residual(data, th, NAIVE)
    per = corpus_residual_fraction(data, th, NAIVE)
    assert g.fraction_per_target == pytest.approx(per.fraction, rel=1e-9)
    assert g.fraction_per_target <= g.fraction + 1e-9
    assert 0.0 <= g.fraction <= 1.0 + 1e-9


def test_a_shared_nuisance_tilt_is_inside_the_global_tangent_space(data):
    """The proof of Proposition 2: one phi shift, one coefficient vector, zero residual."""
    th = np.array([0.0, 0.1, 0.9])
    shift = np.array([0.3, -0.2])
    h_list = _h_from_nuisance_shift(data, th, NAIVE, shift)
    g = global_residual(data, th, NAIVE, h_fn=lambda tgt, q: h_list[tgt.index])
    assert g.fraction < 1e-4  # the ridge in the solve leaves this much
    assert np.allclose(g.coefficients, shift, atol=1e-6)
    assert abs(g.inner_eff) < 1e-6 * max(1.0, g.info_eff)


def test_a_target_varying_tilt_is_absorbed_per_target_but_not_globally(data):
    """Temperature that changes with context length lives in every B_t and outside T."""
    th = np.array([0.0, 0.1, 0.9])
    rng = np.random.default_rng(7)
    h_list = []
    for t in data:
        fit = evaluate_target(t, th, NAIVE, data.M)
        h_list.append(centre(fit.scores[:, 1:] @ rng.normal(size=2), fit.q))
    g = global_residual(data, th, NAIVE, h_fn=lambda tgt, q: h_list[tgt.index])
    assert g.fraction_per_target < 1e-7
    assert g.fraction > 0.3


def test_lexical_tilt_is_inside_the_repaired_tangent_space(data):
    """Prediction 5: a frequency tilt is a shared kappa shift for the repaired estimator."""
    th_r = np.concatenate([[0.0, 0.1, 0.9], np.zeros(data.M + 1)])
    h_list = []
    for t in data:
        fit = evaluate_target(t, th_r, REPAIRED, data.M)
        h_list.append(centre(0.4 * t.f[:, 0] if t.f.ndim == 2 else 0.4 * t.f, fit.q))
    g_r = global_residual(data, th_r, REPAIRED, h_fn=lambda tgt, q: h_list[tgt.index])
    g_n = global_residual(data, np.array([0.0, 0.1, 0.9]), NAIVE,
                          h_fn=lambda tgt, q: h_list[tgt.index])
    assert g_r.fraction < 1e-4
    assert g_n.fraction > g_r.fraction + 0.1


def test_split_half_debiasing_kills_pure_noise(data):
    """Under the fitted null the mismatch is sampling noise, and the cross product says so."""
    th = np.array([0.0, 0.1, 0.9])
    from conftest import draw
    null = draw(make_corpus(n_targets=40, n_clusters=8, seed=41), th, NAIVE, n_per_target=60, seed=5)
    from lcsa.fitting import fit_constrained
    th0 = fit_constrained(null, NAIVE, delta0=0.0, n_starts=1, seed=0).theta
    raw = global_residual(null, th0, NAIVE)
    sh = split_half_residual(null, th0, NAIVE, n_splits=10, seed=1)
    assert raw.fraction > 0.9
    assert sh["signal_share_of_norm2"] < 0.2
    assert sh["norm2_perp_debiased"] < 0.2 * raw.norm2_perp


def test_split_half_debiasing_keeps_real_decay(data):
    th0 = np.array([0.0, 0.1, 0.9])
    sh = split_half_residual(data, th0, NAIVE, n_splits=10, seed=2)
    assert sh["fraction_debiased"] > 0.5
    assert sh["norm2_total_debiased"] > 0.0


def test_implied_bias_is_positive_under_real_decay(data):
    th0 = np.array([0.0, 0.1, 0.9])
    ib = implied_bias(data, th0, NAIVE, n_boot=30, seed=3)
    assert ib["delta_first_order"] > 0.0
    assert ib["delta_first_order_lo"] <= ib["delta_first_order"] <= ib["delta_first_order_hi"]
    assert 0.0 < ib["alignment"] <= 1.0
    assert ib["n_boot_usable"] == 30


def test_lambda_curvature_leak_is_second_order_small(data):
    th0 = np.array([0.0, 0.1, 0.9])
    lk = lambda_curvature_leak(data, th0, NAIVE, a_lambda=0.05)
    assert np.isfinite(lk["delta_leak_second_order"])
    assert abs(lk["delta_leak_second_order"]) < abs(lk["delta_first_order"])
    quad = lambda_curvature_leak(data, th0, NAIVE, a_lambda=0.10)
    assert quad["delta_leak_second_order"] == pytest.approx(4 * lk["delta_leak_second_order"], rel=1e-9)


def test_explained_share_is_one_when_the_direction_is_the_mismatch(data):
    th0 = np.array([0.0, 0.1, 0.9])
    dirs = {}
    for t in data:
        fit = evaluate_target(t, th0, NAIVE, data.M)
        p_emp = (t.n + 0.5) / (t.N + 0.5 * t.V)
        dirs.setdefault("SELF", [None] * len(data))[t.index] = np.log(p_emp) - np.log(fit.q)
    es = explained_share(data, th0, NAIVE, dirs)
    assert es["share_explained"] == pytest.approx(1.0, abs=1e-6)
