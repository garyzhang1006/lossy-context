"""Proposition 1: the mixture, the Abel form, and the simulator all agree."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import brute_marginal, make_corpus

from lcsa.kernels import (POWER, displacements, marginalise, marginalise_abel,
                          marginalise_and_grad, retention, truncation_weights)


def _block(K=9, V=7, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(V), size=K + 1)
    return P


@pytest.mark.parametrize("delta", [0.0, 0.1, 0.316, 1.0, 2.5])
def test_mixture_matches_the_definition(delta):
    P = _block()
    assert marginalise(P, delta) == pytest.approx(brute_marginal(P, delta), abs=1e-14)


@pytest.mark.parametrize("delta", [0.05, 0.316, 1.7])
def test_abel_form_matches_the_atom_form(delta):
    """p = P[0] + sum_j r(j) D_j, the form that avoids forming the atoms at all."""
    P = _block(K=15, V=11, seed=2)
    a = marginalise(P, delta)
    b = marginalise_abel(P, delta)
    assert np.max(np.abs(a - b)) < 1e-14


def test_marginal_is_a_distribution():
    P = _block(K=20, V=13, seed=3)
    for delta in (0.0, 0.2, 5.0):
        p = marginalise(P, delta)
        assert np.all(p > 0)
        assert p.sum() == pytest.approx(1.0, abs=1e-12)


def test_row_semantics_are_the_ones_the_paper_uses():
    """Row 0 is the empty context and row K the full one, and delta selects between them."""
    P = _block(K=6, V=5, seed=4)
    assert marginalise(P, 0.0) == pytest.approx(P[-1], abs=1e-12)
    assert np.max(np.abs(marginalise(P, 40.0) - P[0])) < 1e-6


def test_mixture_matches_a_simulation_of_graded_truncation():
    """Draw one uniform per context and keep the k nearest words, as the model says."""
    P = _block(K=8, V=6, seed=5)
    K = P.shape[0] - 1
    delta = 0.4
    rng = np.random.default_rng(0)
    U = rng.random(400_000)
    r = np.array([retention(np.array([d]), delta)[0] for d in range(K + 2)])
    r[0] = 1.0
    r[K + 1] = 0.0
    # k = max{d : r(d) > U}, with k = 0 when even the nearest word is dropped.
    k = (r[None, 1:K + 1] > U[:, None]).sum(axis=1)
    emp = P[k].mean(axis=0)
    exact = marginalise(P, delta)
    assert np.max(np.abs(emp - exact)) < 4e-3
    counts = np.bincount(k, minlength=K + 1) / U.size
    assert np.max(np.abs(counts - truncation_weights(K, delta))) < 4e-3


@pytest.mark.parametrize("delta", [0.02, 0.3, 1.2])
def test_gradient_matches_finite_difference(delta):
    P = _block(K=12, V=9, seed=6)
    _, A = marginalise_and_grad(P, delta)
    eps = 1e-6
    fd = (marginalise(P, delta + eps) - marginalise(P, delta - eps)) / (2 * eps)
    assert np.max(np.abs(A - fd)) < 1e-7


def test_gradient_sums_to_zero():
    """Every mixture is normalised, so the derivative moves mass without creating it."""
    _, A = marginalise_and_grad(_block(K=10, V=8, seed=7), 0.5)
    assert abs(A.sum()) < 1e-12


def test_precomputed_displacements_change_nothing():
    P = _block(K=14, V=10, seed=8)
    D = displacements(P)
    p1, a1 = marginalise_and_grad(P, 0.37)
    p2, a2 = marginalise_and_grad(P, 0.37, D=D)
    assert np.max(np.abs(p1 - p2)) < 1e-15
    assert np.max(np.abs(a1 - a2)) < 1e-15
    assert D.shape == (P.shape[0] - 1, P.shape[1])
    assert D[0] == pytest.approx(P[1] - P[0])


def test_gradient_vanishes_when_the_cache_is_flat():
    """With no ablation displacement there is nothing for delta to move, so the score is zero."""
    row = np.full(6, 1 / 6)
    P = np.tile(row, (9, 1))
    _, A = marginalise_and_grad(P, 0.3)
    assert np.max(np.abs(A)) < 1e-15


def test_depth_zero_target_is_handled():
    """A passage-initial word has K = 0, one row, and no free information about delta."""
    P = np.array([[0.2, 0.3, 0.5]])
    for delta in (0.0, 1.0, 9.0):
        assert marginalise(P, delta) == pytest.approx(P[0])
    _, A = marginalise_and_grad(P, 0.6)
    assert np.max(np.abs(A)) == 0.0


def test_corpus_targets_carry_consistent_displacements():
    c = make_corpus(n_targets=5, seed=9)
    for t in c:
        assert t.D.shape == (t.K, t.V)
        assert np.max(np.abs(t.D - displacements(t.P))) < 1e-12
