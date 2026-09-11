"""Bound reporting, the local grid, and the profile region's bookkeeping.

The profile tests drive :func:`profile_interval` from a precomputed curve, so
they exercise the region logic itself rather than the optimiser that produced it.
"""

from __future__ import annotations

import numpy as np
import pytest

from lcsa.fitting import _DELTA_MAX, fit, local_grid, profile_interval
from lcsa.kernels import LINEAR
from lcsa.likelihood import NAIVE, loglik_and_grad


def test_linear_kernel_has_a_delta_gradient_at_zero(fitted_corpus):
    """A fit started at ``delta = 0`` under the linear kernel must see a slope."""
    theta = np.array([0.0, 0.1, 0.9])
    _, g = loglik_and_grad(fitted_corpus, theta, NAIVE, LINEAR)
    assert abs(g[0]) > 1e-6


def test_at_bound_flags_a_fit_pinned_at_zero(null_corpus):
    """Counts drawn at exactly zero decay land on the lower bound, which counts."""
    f = fit(null_corpus, NAIVE, n_starts=3, seed=0)
    assert f.delta < 1e-4
    assert f.at_bound


def test_at_bound_flags_the_upper_bound(fitted_corpus):
    f = fit(fitted_corpus, NAIVE, n_starts=1, seed=0, delta_max=0.05)
    assert f.delta == pytest.approx(0.05)
    assert f.at_bound


def test_at_bound_is_false_for_an_interior_fit(fitted_corpus):
    f = fit(fitted_corpus, NAIVE, n_starts=3, seed=0)
    assert 1e-4 < f.delta < _DELTA_MAX - 1e-4
    assert not f.at_bound


def test_at_bound_is_false_when_delta_is_pinned(fitted_corpus):
    """A pinned delta is not an estimate, so it cannot be an estimate at a bound."""
    f = fit(fitted_corpus, NAIVE, fixed={0: 0.0}, n_starts=1, seed=0)
    assert not f.at_bound


def test_local_grid_is_clipped_to_the_parameter_box():
    g = local_grid(4.8, se=0.5, n=9)
    assert g.max() <= _DELTA_MAX + 1e-12
    assert g.min() == 0.0
    assert np.all(np.diff(g) > 0)
    assert np.isclose(g, 4.8).any()


def test_local_grid_multiplicative_fallback_is_clipped_too():
    g = local_grid(2.0, se=None, n=9, span=4.0)
    assert g.max() <= _DELTA_MAX + 1e-12
    assert np.all(np.diff(g) > 0)


def test_profile_region_survives_a_fitted_maximum_above_the_grid():
    """The grid maximum can sit well below the unconstrained fit on a coarse grid."""
    g = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    vals = np.array([-14.0, -11.0, -10.0, -11.0, -14.0])
    reg = profile_interval(None, None, curve=(g, vals), max_loglik=-5.0)
    assert reg.max_loglik == pytest.approx(-5.0)
    assert reg.lo <= 0.2 <= reg.hi


def test_profile_region_is_the_component_containing_the_maximum():
    """A far grid point back under the threshold must not be swept into the region."""
    g = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    vals = np.array([-0.2, 0.0, -5.0, -0.1, -0.2])
    reg = profile_interval(None, None, curve=(g, vals))
    assert reg.lo <= 1.0 <= reg.hi
    assert reg.hi < 2.0
    assert not reg.unbounded_hi
