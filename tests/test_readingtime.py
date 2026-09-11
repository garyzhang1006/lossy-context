"""The held-out reading-time gain: folds, estimators, and the per-word denominator."""

from __future__ import annotations

import numpy as np
import pytest

from lcsa import readingtime
from lcsa.readingtime import reading_time_gain, spillover


def _rt_data(sizes, seed=0):
    """Gaze, two surprisals and two controls over passages of the given lengths."""
    rng = np.random.default_rng(seed)
    passage = np.concatenate([np.full(n, i) for i, n in enumerate(sizes)])
    n = passage.size
    full = rng.normal(size=n) + 5.0
    lossy = full + 0.4 * rng.normal(size=n)
    ctrl = np.column_stack([rng.normal(size=n), rng.random(size=n)])
    gaze = 200.0 + 8.0 * full + 5.0 * lossy + 3.0 * ctrl[:, 0] + rng.normal(size=n)
    return gaze, full, lossy, ctrl, passage


def _stub_fit_ll(monkeypatch, n_base_cols):
    """A MixedLM that succeeds only for the baseline design, as a real one can.

    The mixed fit returns a visibly different beta, so a fold that mixed the two
    estimators leaves a trace in the held-out log-likelihood.
    """
    def fake(y, X, groups, use_mixed):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = max(X.shape[0] - X.shape[1], 1)
        sigma2 = max(float(resid @ resid / dof), 1e-12)
        if use_mixed and X.shape[1] == n_base_cols:
            return 0.5 * beta, sigma2, "MixedLM", True
        return beta, sigma2, "OLS", True

    monkeypatch.setattr(readingtime, "_fit_ll", fake)


def test_a_fold_never_mixes_two_estimators_across_the_arms(monkeypatch):
    """If the comparison arm degrades to OLS, the baseline arm must degrade too."""
    gaze, full, lossy, ctrl, passage = _rt_data([30] * 5)
    _stub_fit_ll(monkeypatch, n_base_cols=5)
    mixed = reading_time_gain(gaze, full, lossy, ctrl, passage, n_folds=5, use_mixed=True)
    plain = reading_time_gain(gaze, full, lossy, ctrl, passage, n_folds=5, use_mixed=False)
    assert mixed.baseline_ll == pytest.approx(plain.baseline_ll)
    assert mixed.full_ll == pytest.approx(plain.full_ll)
    assert mixed.estimator == "OLS"


def test_the_estimator_actually_used_is_reported(monkeypatch):
    gaze, full, lossy, ctrl, passage = _rt_data([30] * 5)

    def fake(y, X, groups, use_mixed):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta, 1.0, "MixedLM" if use_mixed else "OLS", True

    monkeypatch.setattr(readingtime, "_fit_ll", fake)
    r = reading_time_gain(gaze, full, lossy, ctrl, passage, n_folds=5, use_mixed=True)
    assert r.estimator == "MixedLM"


def test_gain_per_word_counts_only_the_words_it_evaluated():
    """A fold too small to train on is skipped, and its words are never scored."""
    sizes = [100, 2, 2, 2]
    gaze, full, lossy, ctrl, passage = _rt_data(sizes, seed=3)
    r = reading_time_gain(gaze, full, lossy, ctrl, passage, n_folds=4, use_mixed=False)
    # Each passage loses its first word to the spillover lag, so only the three
    # one-word passages are ever held out; the big one leaves three training
    # rows, too few for the design.
    assert r.n_words == 3
    assert r.gain_per_word == pytest.approx(r.full_ll - r.baseline_ll)
    assert "skipped" in r.note


def test_spillover_without_positions_lags_the_retained_array():
    x = np.array([1.0, 2.0, 3.0, 10.0])
    passage = np.array([0, 0, 0, 1])
    out = spillover(x, passage)
    assert np.isnan(out[0]) and np.isnan(out[3])
    assert out[1] == 1.0 and out[2] == 2.0


def test_spillover_by_position_refuses_to_invent_a_previous_word():
    """With word numbers given, a dropped word makes its successor's lag missing."""
    x = np.array([1.0, 2.0, 3.0])
    passage = np.array([0, 0, 0])
    position = np.array([1, 3, 4])  # word 2 was dropped upstream
    out = spillover(x, passage, position=position)
    assert np.isnan(out[0])
    assert np.isnan(out[1])
    assert out[2] == 2.0


def test_spillover_by_position_is_not_fooled_by_row_order():
    x = np.array([3.0, 1.0, 2.0])
    passage = np.array([0, 0, 0])
    position = np.array([3, 1, 2])
    out = spillover(x, passage, position=position)
    assert np.isnan(out[1])
    assert out[2] == 1.0
    assert out[0] == 2.0
