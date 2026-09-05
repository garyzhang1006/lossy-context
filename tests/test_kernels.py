"""The retention kernel and its truncation weights."""

from __future__ import annotations

import numpy as np
import pytest

from lcsa.kernels import (LINEAR, POWER, d_half_from_delta, d_retention_d_delta,
                          d_truncation_weights, delta_from_d_half, get_kernel,
                          retention, truncation_weights)


@pytest.mark.parametrize("kernel", [POWER, LINEAR])
@pytest.mark.parametrize("delta", [0.0, 0.05, 0.316, 1.0, 3.0])
@pytest.mark.parametrize("K", [0, 1, 5, 32])
def test_weights_are_a_distribution(kernel, delta, K):
    w = truncation_weights(K, delta, kernel)
    assert w.shape == (K + 1,)
    assert np.all(w >= -1e-15)
    assert abs(w.sum() - 1.0) < 1e-12


@pytest.mark.parametrize("delta", [0.05, 0.316, 1.0])
def test_tail_sum_equals_retention(delta):
    """Sum of atoms from m upward is r(m), which is what makes the mixture exact."""
    K = 20
    w = truncation_weights(K, delta, POWER)
    for m in range(1, K + 1):
        assert w[m:].sum() == pytest.approx(retention(np.array([m]), delta)[0], abs=1e-12)


def test_zero_delta_is_full_retention():
    """delta = 0 must put all mass on the deepest row, which is full context."""
    w = truncation_weights(12, 0.0, POWER)
    assert w[-1] == pytest.approx(1.0)
    assert w[:-1] == pytest.approx(np.zeros(12))


def test_large_delta_concentrates_on_empty_context():
    w = truncation_weights(12, 8.0, POWER)
    assert w[0] > 0.99


def test_retention_is_monotone_in_distance_and_delta():
    d = np.arange(1, 20)
    r = retention(d, 0.4)
    assert np.all(np.diff(r) < 0)
    assert np.all(retention(d, 0.8) < r)
    assert retention(np.array([0]), 0.9)[0] == pytest.approx(1.0)


@pytest.mark.parametrize("kernel", [POWER, LINEAR])
@pytest.mark.parametrize("delta", [0.05, 0.3, 0.9])
def test_retention_gradient_matches_finite_difference(kernel, delta):
    # The linear kernel has a kink where it clips at zero, so it is differenced
    # only strictly inside its active region; the power kernel is smooth
    # everywhere and is differenced out to distance 24.
    d = np.arange(0, 25) if kernel is POWER else np.arange(0, int(0.9 / delta) + 1)
    eps = 1e-8 if kernel is LINEAR else 1e-6
    fd = (kernel(d, delta + eps) - kernel(d, delta - eps)) / (2 * eps)
    assert np.max(np.abs(d_retention_d_delta(d, delta, kernel) - fd)) < 1e-5


def test_linear_kernel_gradient_is_zero_past_the_clip():
    """Past ``d = 1/delta`` the linear kernel is flat at zero and carries no signal."""
    assert d_retention_d_delta(np.array([30.0, 40.0]), 0.05, LINEAR) == pytest.approx([0, 0])


@pytest.mark.parametrize("delta", [0.02, 0.3, 1.4])
def test_weight_gradient_matches_finite_difference(delta):
    K, eps = 15, 1e-6
    fd = (truncation_weights(K, delta + eps) - truncation_weights(K, delta - eps)) / (2 * eps)
    assert np.max(np.abs(d_truncation_weights(K, delta) - fd)) < 1e-6


def test_weight_gradient_sums_to_zero():
    """The atoms always sum to one, so their derivative must sum to zero."""
    assert abs(d_truncation_weights(19, 0.44).sum()) < 1e-12


@pytest.mark.parametrize("d_half", [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
def test_d_half_round_trip(d_half):
    delta = delta_from_d_half(d_half)
    assert d_half_from_delta(delta) == pytest.approx(d_half, rel=1e-9)
    assert retention(np.array([d_half]), delta)[0] == pytest.approx(0.5, rel=1e-9)


def test_infinite_d_half_is_zero_delta():
    assert delta_from_d_half(float("inf")) == 0.0
    assert np.isinf(d_half_from_delta(0.0))


def test_get_kernel_accepts_name_or_object_and_rejects_junk():
    assert get_kernel("power") is POWER
    assert get_kernel(POWER) is POWER
    with pytest.raises(ValueError, match="unknown kernel"):
        get_kernel("exponential-decay")


def test_negative_delta_is_refused():
    """A negative delta means retention rising with distance, which is not in the family."""
    with pytest.raises(ValueError):
        truncation_weights(5, -0.1)
