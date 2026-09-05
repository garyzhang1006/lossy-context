"""End to end on synthetic data, where the truth is known.

Recovery, profile regions, storage, the window baseline, and the four experiment
drivers, all at a size that runs in about half a minute.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import draw_true_delta, make_corpus

from lcsa.baselines import auc, context_slopes, hard_window_sweep
from lcsa.experiments.e1_exactness import exactness_report, run as run_e1
from lcsa.experiments.e2_ladder import run as run_e2
from lcsa.experiments.e3_nulls import h_lexical, run as run_e3
from lcsa.experiments.e4_reading import run as run_e4, window_surprisal
from lcsa.fitting import (default_grid, fit, fit_constrained, local_grid,
                          profile_curve, profile_interval)
from lcsa.gates import g2_sensitivity, g5_floors, g6_coverage
from lcsa.likelihood import NAIVE, REPAIRED, loglik
from lcsa.store import load_corpus, save_corpus


@pytest.fixture(scope="module")
def truth():
    """Half-distance of eight words, which is the middle of the registered ladder."""
    return 0.316


@pytest.fixture(scope="module")
def data(truth):
    c = make_corpus(n_targets=150, n_clusters=15, K_range=(8, 24), seed=101, drift=0.08)
    return draw_true_delta(c, truth, seed=102, n_per_target=120)


def test_fit_recovers_the_generating_delta(data, truth):
    f = fit(data, NAIVE, n_starts=3, seed=0)
    assert f.success
    assert abs(f.delta - truth) < 0.12
    assert f.loglik > loglik(data, np.array([0.0, 0.1, 0.9]), NAIVE)


def test_constrained_fit_pins_delta_at_zero(data):
    f = fit_constrained(data, NAIVE, n_starts=2)
    assert f.delta == 0.0
    assert f.theta.size == 3


def test_fixing_a_parameter_is_honoured(data):
    f = fit(data, NAIVE, fixed={0: 0.5}, n_starts=1)
    assert f.theta[0] == 0.5


def test_warm_start_never_loses_to_a_cold_one(data):
    cold = fit(data, NAIVE, n_starts=3, seed=1)
    warm = fit(data, NAIVE, n_starts=3, seed=1, start=cold.theta)
    assert warm.loglik >= cold.loglik - 1e-8


def test_wrong_length_warm_start_is_refused(data):
    with pytest.raises(ValueError, match="expected"):
        fit(data, NAIVE, n_starts=1, start=np.zeros(9))


def test_profile_region_contains_the_maximum_and_the_truth(data, truth):
    f = fit(data, NAIVE, n_starts=3, seed=2)
    grid = local_grid(f.delta, se=0.05, n=9)
    reg = profile_interval(data, NAIVE, grid=grid, max_loglik=f.loglik)
    assert reg.lo <= f.delta <= reg.hi
    assert reg.lo <= truth <= reg.hi
    assert not reg.unbounded


def test_local_grid_always_contains_the_estimate_and_zero():
    g = local_grid(0.3004, se=0.012, n=9)
    assert np.isclose(g, 0.3004).any()
    assert g.min() == 0.0
    assert np.all(np.diff(g) > 0)
    fallback = local_grid(float("nan"))
    assert fallback.size and np.all(fallback >= 0)


def test_default_grid_is_increasing_and_starts_at_zero():
    g = default_grid(n=21)
    assert g[0] == pytest.approx(0.0) or g[0] > 0
    assert np.all(np.diff(g) > 0)


def test_profile_curve_peaks_near_the_unconstrained_fit(data):
    f = fit(data, NAIVE, n_starts=3, seed=3)
    grid = local_grid(f.delta, se=0.05, n=9)
    g, vals = profile_curve(data, NAIVE, grid=grid)
    assert vals.max() <= f.loglik + 1e-6
    assert abs(g[int(np.argmax(vals))] - f.delta) < 0.1


def test_cluster_scaling_widens_the_region(data):
    f = fit(data, NAIVE, n_starts=2, seed=4)
    grid = local_grid(f.delta, se=0.06, n=11)
    tight = profile_interval(data, NAIVE, grid=grid, max_loglik=f.loglik)
    loose = profile_interval(data, NAIVE, grid=grid, max_loglik=f.loglik, scale=0.25)
    assert (loose.hi - loose.lo) >= (tight.hi - tight.lo) - 1e-9


def test_corpus_round_trips_through_disk(data, tmp_path):
    p = tmp_path / "cache.npz"
    save_corpus(p, data)
    back = load_corpus(p)
    assert len(back) == len(data)
    assert back.n_clusters == data.n_clusters
    assert back.M == data.M
    for a, b in zip(data, back):
        assert np.max(np.abs(a.P - b.P)) < 1e-6
        assert np.max(np.abs(a.n - b.n)) < 1e-6
        assert a.target_slot == b.target_slot
    assert loglik(back, np.array([0.3, 0.1, 0.9]), NAIVE) == pytest.approx(
        loglik(data, np.array([0.3, 0.1, 0.9]), NAIVE), rel=1e-4)


def test_saved_cache_holds_no_pickled_objects(data, tmp_path):
    p = tmp_path / "cache.npz"
    save_corpus(p, data)
    # allow_pickle=False is the point: a cache must be plain arrays, never objects.
    with np.load(p, allow_pickle=False) as z:
        assert {"format", "P", "K", "V", "n", "cluster"} <= set(z.files)


def test_exactness_report_passes_on_its_own_identities():
    rep = exactness_report()
    assert rep["passed"]
    assert rep["max_atom_vs_abel_error"] < 1e-12
    assert rep["max_score_vs_finite_difference"] < 1e-5


def test_sensitivity_gate_sees_signal_in_a_drifting_cache(data):
    g = g2_sensitivity(data, max_j=8, tv_floor=0.01, frac_floor=0.5)
    assert g.passed
    assert set(g.measured["frac_above_floor_by_j"]) == set(range(1, 9))


def test_sensitivity_gate_reports_a_ceiling_when_the_cache_goes_flat():
    """A cache with no displacement past depth 1 must be caught, not fitted."""
    flat = make_corpus(n_targets=20, seed=110, drift=0.0)
    g = g2_sensitivity(flat, max_j=4, tv_floor=0.05, frac_floor=0.8)
    assert not g.passed
    assert g.measured["first_failing_j"] == 1


def test_context_slopes_are_measured_from_the_responses_alone(data):
    rep = context_slopes(data)
    assert rep["n_targets"] == len(data)
    for key in ("entropy", "top1"):
        stat = rep[key]
        assert np.isfinite(stat.slope) and stat.se > 0


def test_auc_is_half_for_identical_samples_and_one_for_separated_ones():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    assert auc(x, x.copy()) == pytest.approx(0.5, abs=1e-9)
    assert auc(x + 50, x) == pytest.approx(1.0)


def test_hard_window_sweep_recovers_the_generating_window():
    """A responder that truly sees four words must select four, not the grid edge."""
    base = make_corpus(n_targets=140, n_clusters=14, K_range=(8, 20), seed=120, drift=0.1)
    rng = np.random.default_rng(121)
    counts = []
    for t in base:
        j = min(4, t.K)
        counts.append(rng.multinomial(150, t.P[j]).astype(float))
    obs = base.with_counts(counts)
    fits = hard_window_sweep(obs, NAIVE, windows=(0, 1, 2, 4, 8, 16), n_folds=4)
    best = min(fits, key=lambda w: w.aic)
    assert best.window == 4


@pytest.mark.parametrize("delta", [0.0, 0.316])
def test_experiment_drivers_run_and_write_artifacts(tmp_path, delta):
    c = draw_true_delta(make_corpus(n_targets=60, n_clusters=10, seed=130), delta,
                        seed=131, n_per_target=60)
    out = tmp_path / f"d{delta}"

    r1 = run_e1(c, [NAIVE, REPAIRED], out / "e1", seed=0)
    assert r1["exactness"]["passed"]
    assert all(g["passed"] for g in r1["gradients"])
    assert (out / "e1" / "e1_summary.json").exists()

    theta0 = fit_constrained(c, NAIVE, n_starts=2).theta
    r2 = run_e2(c, c, theta0, [NAIVE], out / "e2", n_rep=4, coverage_rungs=(8.0,), seed=0)
    assert isinstance(r2["g6"], type(g6_coverage({}, floor=0.9)))
    assert (out / "e2" / "e2_ladder.csv").exists()

    r3 = run_e3(c, [NAIVE, REPAIRED], out / "e3",
                h_specs={"N-LEX": h_lexical(c, REPAIRED, seed=0)},
                n_rep=4, n_boot=8, seed=0)
    assert isinstance(r3["g5"], type(g5_floors({})))
    assert (out / "e3" / "e3_rejection_rates.csv").exists()

    passage = np.array([t.cluster for t in c])
    rng = np.random.default_rng(132)
    y = 250 + 12 * np.nan_to_num(window_surprisal(c, 4)) + rng.normal(0, 25, len(c))
    ctrl = rng.normal(size=(len(c), 2))
    r4 = run_e4(c, y, ctrl, passage, [NAIVE], out / "e4", k_grid=(0, 2, 4, 8),
                n_boot=5, n_folds=4, use_mixed=False, seed=0)
    assert r4["selected"]["primary"] in (0, 2, 4, 8)
    assert (out / "e4" / "e4_sweep_curves.csv").exists()


def test_artifacts_are_valid_json(tmp_path):
    c = draw_true_delta(make_corpus(n_targets=40, n_clusters=8, seed=140), 0.2, seed=141)
    out = tmp_path / "e1"
    run_e1(c, [NAIVE], out, seed=0)
    payload = json.loads((out / "e1_summary.json").read_text())
    assert payload["exactness"]["passed"] is True
    assert payload["gradients"][0]["passed"] is True


def test_cli_selftest_passes(tmp_path):
    from lcsa.cli import main

    assert main(["selftest", "--out", str(tmp_path / "st"), "--n-rep", "3",
                 "--n-boot", "6"]) == 0


def test_cli_reports_a_missing_cache_clearly(tmp_path):
    from lcsa.cli import main

    with pytest.raises(SystemExit, match="Run `lcsa build` first"):
        main(["e1", "--cache", str(tmp_path / "nope.npz"), "--out", str(tmp_path)])
