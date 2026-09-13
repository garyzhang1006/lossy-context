"""Reader generation and the calibration searches."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_corpus

from lcsa import readers
from lcsa.kernels import LINEAR, POWER, retention
from lcsa.likelihood import NAIVE


def _capture_delta(monkeypatch):
    """Record every delta ``reader_ladder`` hands to the truncation weights."""
    seen = []
    real = readers.truncation_weights

    def spy(K, delta, kernel=POWER):
        seen.append(float(delta))
        return real(K, delta, kernel)

    monkeypatch.setattr(readers, "truncation_weights", spy)
    return seen


@pytest.mark.parametrize("kernel", [POWER, LINEAR])
def test_reader_ladder_generates_at_the_rung_of_its_own_kernel(monkeypatch, kernel):
    """``r(d_half) = 1/2`` must hold under the kernel the mask is drawn with."""
    corpus = make_corpus(n_targets=8, n_clusters=2, seed=5)
    corpus = corpus.with_counts([np.full(t.V, 4.0) for t in corpus])
    seen = _capture_delta(monkeypatch)
    readers.reader_ladder(corpus, 8.0, np.array([0.0, 0.1, 0.9]), NAIVE, seed=0,
                          kernel=kernel)
    assert seen
    delta = seen[0]
    assert retention(np.array([8.0]), delta, kernel)[0] == pytest.approx(0.5, rel=1e-9)


def test_ladder_deltas_follow_the_kernel():
    power = readers.ladder_deltas()
    linear = readers.ladder_deltas(LINEAR)
    assert power[-1] == 0.0 and linear[-1] == 0.0
    for d_half, dp, dl in zip(readers.LADDER_DHALF[:-1], power, linear):
        assert retention(np.array([d_half]), dp, POWER)[0] == pytest.approx(0.5, rel=1e-9)
        assert retention(np.array([d_half]), dl, LINEAR)[0] == pytest.approx(0.5, rel=1e-9)


def _counts(corpus):
    return [np.asarray(t.n) for t in corpus]


def test_n0_prime_carries_its_target_level_tilt_by_default():
    """The registered design gives N0-prime a passage tilt *and* a smaller
    target-level one; the second used to default to zero, so the only caller
    passing its scale positionally never ran it."""
    corpus = make_corpus(n_targets=24, n_clusters=6, seed=9)
    corpus = corpus.with_counts([np.full(t.V, 8.0) for t in corpus])
    theta0 = np.array([0.0, 0.1, 0.9])
    sigma = 0.8

    assert 0.0 < readers.TARGET_TILT_RATIO < 1.0
    default = readers.reader_n0_prime(corpus, theta0, NAIVE, sigma, seed=3)
    tied = readers.reader_n0_prime(corpus, theta0, NAIVE, sigma, seed=3,
                                   sigma_target=readers.TARGET_TILT_RATIO * sigma)
    off = readers.reader_n0_prime(corpus, theta0, NAIVE, sigma, seed=3, sigma_target=0.0)

    for a, b in zip(_counts(default), _counts(tied)):
        assert np.array_equal(a, b)
    assert any(not np.array_equal(a, b) for a, b in zip(_counts(default), _counts(off)))


def test_the_design_effect_bisection_still_converges_with_the_target_tilt():
    """The target-level component is tied to the passage scale precisely so that
    the single bisection on ``sigma_passage`` still calibrates the design."""
    corpus = make_corpus(n_targets=40, n_clusters=8, seed=10)
    corpus = corpus.with_counts([np.full(t.V, 12.0) for t in corpus])
    cal = readers.calibrate_n0_prime(corpus, np.array([0.0, 0.1, 0.9]), NAIVE,
                                     target_design_effect=2.5, seed=4, n_rep=2,
                                     n_iter=8, tol=0.1)
    assert cal["converged"] is True
    assert cal["at_bound"] is False
    assert cal["design_effect"] == pytest.approx(2.5, rel=0.1)


def _stub_calibration_search(monkeypatch, table):
    """Drive the bisection off a deterministic design effect per tilt scale.

    ``reader_n0_prime`` is replaced by the scale itself, which is all the stubbed
    score test needs, so the search runs without drawing a single corpus.
    """
    from types import SimpleNamespace

    import lcsa.inference as inference

    monkeypatch.setattr(readers, "reader_n0_prime",
                        lambda corpus, theta0, model, sigma_passage, **kw: float(sigma_passage))
    monkeypatch.setattr(inference, "score_test",
                        lambda c, *a, **kw: SimpleNamespace(design_effect=table(float(c))))


def test_calibration_never_returns_a_scale_whose_draw_failed(monkeypatch):
    """Every mid above 1.0 evaluates to nan, so the last mid is not a usable answer."""
    def table(s):
        if s >= 2.0:
            return 10.0
        if s > 1.0:
            return float("nan")
        return 1.0 + s

    _stub_calibration_search(monkeypatch, table)
    corpus = make_corpus(n_targets=4, n_clusters=2, seed=6)
    cal = readers.calibrate_n0_prime(corpus, np.array([0.0, 0.1, 0.9]), NAIVE,
                                     target_design_effect=5.0, n_rep=1)
    assert np.isfinite(cal["design_effect"])
    assert cal["sigma"] == pytest.approx(1.0)
    assert cal["design_effect"] == pytest.approx(2.0)
    assert cal["converged"] is False


def test_calibration_reports_the_scale_that_hit_the_target(monkeypatch):
    _stub_calibration_search(monkeypatch, lambda s: 1.0 + 4.0 * s)
    corpus = make_corpus(n_targets=4, n_clusters=2, seed=6)
    cal = readers.calibrate_n0_prime(corpus, np.array([0.0, 0.1, 0.9]), NAIVE,
                                     target_design_effect=5.0, n_rep=1)
    assert cal["sigma"] == pytest.approx(1.0, abs=1e-3)
    assert cal["design_effect"] == pytest.approx(5.0, rel=0.05)
    assert cal["converged"] is True
    assert cal["at_bound"] is False
