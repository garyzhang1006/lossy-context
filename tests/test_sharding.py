"""Sharded replicate loops reproduce the monolithic run byte for byte.

Every Slurm array task computes a half-open range of replicates and ``lcsa merge``
concatenates them, so the property the cluster pipeline rests on is that the
cut changes nothing.  Each test below runs a leg once in one process and once
as stages plus shards, then compares the registered artifacts as bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import draw_true_delta, make_corpus

from lcsa.experiments import e2_ladder as e2
from lcsa.experiments import e3_nulls as e3
from lcsa.experiments import e4_reading as e4
from lcsa.experiments.shards import read_shards, rep_indices, write_shard
from lcsa.fitting import fit, fit_constrained
from lcsa.inference import cluster_bootstrap
from lcsa.likelihood import NAIVE, REPAIRED
from lcsa.readers import tilt_directions, tilted_cache
from lcsa.store import save_corpus

MODELS = [NAIVE, REPAIRED]


@pytest.fixture(scope="module")
def corpus():
    return draw_true_delta(make_corpus(n_targets=60, n_clusters=10, seed=130), 0.316,
                           seed=131, n_per_target=60)


def _same(a, b, names):
    for f in names:
        assert (a / f).read_bytes() == (b / f).read_bytes(), f


def test_rep_indices_validates_the_range():
    assert rep_indices(5) == range(5)
    assert rep_indices(5, (2, 4)) == range(2, 4)
    with pytest.raises(ValueError):
        rep_indices(5, (3, 3))
    with pytest.raises(ValueError):
        rep_indices(5, (-1, 2))


def test_read_shards_rejects_overlap_gap_and_a_missing_first_shard(tmp_path):
    rows = lambda r: [{"replicate": b} for b in r]  # noqa: E731
    write_shard(tmp_path, "x", range(0, 3), rows(range(0, 3)))
    write_shard(tmp_path, "x", range(3, 5), rows(range(3, 5)))
    assert [r["replicate"] for r in read_shards(tmp_path, "x")] == [0, 1, 2, 3, 4]
    write_shard(tmp_path, "x", range(4, 6), rows(range(4, 6)))
    with pytest.raises(ValueError, match="overlap"):
        read_shards(tmp_path, "x")
    (tmp_path / "shards" / "x_000004-000006.json").unlink()
    write_shard(tmp_path, "x", range(6, 8), rows(range(6, 8)))
    with pytest.raises(ValueError, match="missing"):
        read_shards(tmp_path, "x")
    (tmp_path / "shards" / "x_000000-000003.json").unlink()
    (tmp_path / "shards" / "x_000006-000008.json").unlink()
    with pytest.raises(ValueError, match="not 0"):
        read_shards(tmp_path, "x")
    with pytest.raises(FileNotFoundError):
        read_shards(tmp_path, "y")


def test_shards_round_trip_nan_as_null(tmp_path):
    write_shard(tmp_path, "z", range(0, 1), [{"replicate": 0, "v": float("nan")}])
    v = read_shards(tmp_path, "z")[0]["v"]
    assert isinstance(v, float) and np.isnan(v)


def test_bootstrap_halves_concatenate_to_the_whole(corpus):
    stat = lambda sub: {"n": float(sub.total_n)}  # noqa: E731
    whole = cluster_bootstrap(corpus, stat, n_boot=6, seed=3)
    halves = (cluster_bootstrap(corpus, stat, seed=3, reps=range(0, 2))
              + cluster_bootstrap(corpus, stat, seed=3, reps=range(2, 6)))
    assert whole == halves
    assert [r["replicate"] for r in whole] == list(range(6))


def test_e2_shards_reproduce_the_monolithic_run(corpus, tmp_path):
    theta0 = fit_constrained(corpus, NAIVE, n_starts=2).theta
    mono, sh = tmp_path / "mono", tmp_path / "sh"
    e2.run(corpus, corpus, theta0, MODELS, mono, n_rep=4, coverage_rungs=(8.0,), seed=0)
    e2.run_ladder(corpus, corpus, theta0, MODELS, sh, seed=0)
    e2.run_coverage_shard(corpus, corpus, theta0, MODELS, sh, range(0, 1),
                          coverage_rungs=(8.0,), seed=0)
    e2.run_coverage_shard(corpus, corpus, theta0, MODELS, sh, range(1, 4),
                          coverage_rungs=(8.0,), seed=0)
    e2.merge(sh, MODELS, coverage_rungs=(8.0,))
    _same(mono, sh, ["e2_summary.json", "e2_coverage.csv", "e2_ladder.csv"])
    cov = json.loads((mono / "e2_summary.json").read_text())["coverage"]
    assert all(r["n_usable"] == 4 for r in cov)


def test_e3_shards_reproduce_the_monolithic_run(corpus, tmp_path):
    h = {"N-LEX": e3.h_lexical(corpus, REPAIRED, seed=0)}
    mono, sh = tmp_path / "mono", tmp_path / "sh"
    e3.run(corpus, MODELS, mono, h_specs=h, n_rep=4, n_boot=6, seed=0)
    e3.prepare(corpus, MODELS, sh, h_specs=h, seed=0)
    prep = e3.load_prepared(sh, corpus)
    assert list(prep["readers"]) == ["N0", "N0-PRIME", "N-LEX"]
    for nm in prep["readers"]:
        e3.run_replicate_shard(corpus, prep, MODELS, sh, nm, range(0, 2), seed=0)
        e3.run_replicate_shard(corpus, prep, MODELS, sh, nm, range(2, 4), seed=0)
    e3.run_human(corpus, prep, MODELS, sh, seed=0)
    e3.run_contrast_shard(corpus, prep, MODELS, sh, range(0, 2), seed=0)
    e3.run_contrast_shard(corpus, prep, MODELS, sh, range(2, 6), seed=0)
    e3.merge(sh)
    _same(mono, sh, ["e3_summary.json", "e3_rejection_rates.csv", "e3_nulls.json",
                     "e3_human.json", "e3_human_fit.csv", "e3_calibration.json"])
    rates = json.loads((mono / "e3_summary.json").read_text())["rejection_rates"]
    assert {r["reader"] for r in rates} == {"N0", "N0-PRIME", "N-LEX"}
    assert all(r["n_replicates"] == 4 for r in rates)
    assert all({"reject_cr1", "reject_cr3", "reject_cluster_robust"} <= set(r) for r in rates)


def test_e3_prepared_stage_refuses_a_different_cache(corpus, tmp_path):
    e3.prepare(corpus, [NAIVE], tmp_path, readers=["N0"], seed=0)
    other = make_corpus(n_targets=61, n_clusters=10, seed=7)
    with pytest.raises(ValueError, match="different builds"):
        e3.load_prepared(tmp_path, other)
    prep = e3.load_prepared(tmp_path, corpus)
    with pytest.raises(ValueError, match="primary estimator"):
        e3.run_replicate_shard(corpus, prep, [REPAIRED, NAIVE], tmp_path, "N0", range(1))
    with pytest.raises(KeyError):
        e3.run_replicate_shard(corpus, prep, [NAIVE], tmp_path, "N-LEX", range(1))


def test_e4_shards_reproduce_the_monolithic_run(corpus, tmp_path):
    rng = np.random.default_rng(3)
    passage = np.array([t.cluster for t in corpus])
    y = 250 + 12 * np.nan_to_num(e4.window_surprisal(corpus, 4)) + rng.normal(0, 25, len(corpus))
    ctrl = np.column_stack([rng.normal(size=len(corpus)), rng.normal(size=len(corpus))])
    dirs = tilt_directions(corpus, e3.h_lexical(corpus, REPAIRED), orthogonalise=False)
    refs = {"primary": None, "tilt": tilted_cache(corpus, dirs, 0.3), "self": corpus}
    fitted = {"human": (fit(corpus, NAIVE, n_starts=1, seed=0).theta, NAIVE)}
    kw = dict(references=refs, fitted=fitted, k_grid=(0, 2, 4, 8), n_folds=4,
              use_mixed=False, seed=0)
    mono, sh = tmp_path / "mono", tmp_path / "sh"
    e4.run(corpus, y, ctrl, passage, [NAIVE], mono, n_boot=5, **kw)
    e4.run_sweep(corpus, y, ctrl, passage, [NAIVE], sh, **kw)
    e4.run_argmax_shard(corpus, y, ctrl, passage, sh, range(0, 2), refs, (0, 2, 4, 8), 0)
    e4.run_argmax_shard(corpus, y, ctrl, passage, sh, range(2, 5), refs, (0, 2, 4, 8), 0)
    e4.merge(sh)
    _same(mono, sh, ["e4_summary.json", "e4_sweep_curves.csv", "e4_rt_gain.csv",
                     "e4_hard_window_likelihood.csv"])
    res = json.loads((mono / "e4_summary.json").read_text())
    assert set(res["selected"]) == {"primary", "tilt", "self"}
    assert res["selected"]["self"] == res["selected"]["primary"]
    assert all(b["n_boot"] == 5 for b in res["argmax_bootstrap"].values())


def test_window_surprisal_refuses_a_reference_of_another_size(corpus):
    with pytest.raises(ValueError, match="targets.csv"):
        e4.window_surprisal(corpus, 2, make_corpus(n_targets=5, seed=1))


def test_cli_stages_and_merge_reproduce_the_monolithic_commands(corpus, tmp_path):
    from lcsa.cli import main

    cache = tmp_path / "cache.npz"
    save_corpus(cache, corpus)
    mono, sh = str(tmp_path / "mono"), str(tmp_path / "sh")
    c = ["--cache", str(cache)]
    assert main(["e2", *c, "--out", mono, "--n-rep", "3"]) == 0
    assert main(["e2", *c, "--out", sh, "--stage", "ladder"]) == 0
    assert main(["e2", *c, "--out", sh, "--stage", "coverage", "--n-rep", "3",
                 "--rep-stop", "1"]) == 0
    assert main(["e2", *c, "--out", sh, "--stage", "coverage", "--n-rep", "3",
                 "--rep-start", "1"]) == 0
    assert main(["e3", *c, "--out", mono, "--n-rep", "3", "--n-boot", "4"]) == 0
    assert main(["e3", *c, "--out", sh, "--stage", "prepare"]) == 0
    assert main(["e3", *c, "--out", sh, "--stage", "replicates", "--n-rep", "3",
                 "--rep-stop", "2", "--readers", "N0,N0-PRIME"]) == 0
    assert main(["e3", *c, "--out", sh, "--stage", "replicates", "--n-rep", "3",
                 "--rep-stop", "2", "--readers", "N-LEX"]) == 0
    assert main(["e3", *c, "--out", sh, "--stage", "replicates", "--n-rep", "3",
                 "--rep-start", "2"]) == 0
    assert main(["e3", *c, "--out", sh, "--stage", "human"]) == 0
    assert main(["e3", *c, "--out", sh, "--stage", "contrast", "--n-boot", "4",
                 "--boot-stop", "3"]) == 0
    assert main(["e3", *c, "--out", sh, "--stage", "contrast", "--n-boot", "4",
                 "--boot-start", "3"]) == 0
    assert main(["register", "--out", sh, "--n-rep", "3", "--n-boot", "4", "--legs", "e2,e3",
                 "--readers", "N0", "N0-PRIME", "N-LEX"]) == 0
    assert main(["merge", "--out", sh, "--legs", "e2,e3"]) == 0
    card = json.loads((Path(sh) / "scorecard.json").read_text())
    assert {r["id"] for r in card["predictions"]} == set(range(1, 12))
    assert card["registration_sha256"] == (Path(sh) / "registration.sha256").read_text().split()[0]
    _same(tmp_path / "mono", tmp_path / "sh",
          ["e2_summary.json", "e2_coverage.csv", "e3_summary.json",
           "e3_rejection_rates.csv", "e3_human.json"])


def test_cli_coverage_stage_needs_the_ladder_first(corpus, tmp_path):
    from lcsa.cli import main

    cache = tmp_path / "cache.npz"
    save_corpus(cache, corpus)
    with pytest.raises(SystemExit, match="stage ladder"):
        main(["e2", "--cache", str(cache), "--out", str(tmp_path / "o"), "--stage",
              "coverage", "--rep-stop", "1"])
    with pytest.raises(FileNotFoundError, match="stage prepare"):
        main(["e3", "--cache", str(cache), "--out", str(tmp_path / "o"), "--stage",
              "replicates", "--rep-stop", "1"])


def test_cli_self_reference_run_prepares_the_plain_floor_alone(corpus, tmp_path):
    from lcsa.cli import main

    cache = tmp_path / "cache.npz"
    save_corpus(cache, corpus)
    out = tmp_path / "self"
    assert main(["e3", "--cache", str(cache), "--out", str(out), "--nulls", "none",
                 "--readers", "N0", "--no-human", "--n-rep", "2"]) == 0
    rates = json.loads((out / "e3_summary.json").read_text())["rejection_rates"]
    assert {r["reader"] for r in rates} == {"N0"}
    assert json.loads((out / "e3_summary.json").read_text())["human"] is None


def test_prepared_stage_refuses_another_kernel(null_corpus, tmp_path):
    from lcsa.experiments import e3_nulls as e3
    from lcsa.kernels import LINEAR
    from lcsa.likelihood import NAIVE

    prep = e3.prepare(null_corpus, [NAIVE], tmp_path, None, seed=0, readers=["N0"])
    assert prep["kernel"] == "power"
    loaded = e3.load_prepared(tmp_path, null_corpus)
    assert loaded["kernel"] == "power"
    with pytest.raises(ValueError, match="--kernel power"):
        e3.run_human(null_corpus, loaded, [NAIVE], tmp_path, kernel=LINEAR)
    with pytest.raises(ValueError, match="--kernel power"):
        e3.run_replicate_shard(null_corpus, loaded, [NAIVE], tmp_path, "N0", range(0, 1),
                               kernel=LINEAR)


def test_cli_runs_the_linear_kernel_end_to_end(null_corpus, tmp_path):
    from lcsa.cli import main
    from lcsa.store import save_corpus

    cache = tmp_path / "cache.npz"
    save_corpus(cache, null_corpus)
    out = tmp_path / "lin"
    args = ["--cache", str(cache), "--out", str(out), "--estimators", "naive",
            "--kernel", "linear"]
    assert main(["e3", *args, "--nulls", "none", "--readers", "N0", "--n-rep", "2",
                 "--n-boot", "2"]) == 0
    prep = json.loads((out / "e3_prepared.json").read_text())
    assert prep["kernel"] == "linear"
    assert main(["e2", *args, "--stage", "ladder"]) == 0
    assert json.loads((out / "e2_theta0.json").read_text())["kernel"] == "linear"
    with pytest.raises(SystemExit, match="kernel"):
        main(["e2", "--cache", str(cache), "--out", str(out), "--estimators", "naive",
              "--stage", "coverage", "--n-rep", "1", "--rep-start", "0", "--rep-stop", "1"])


def test_shards_round_trip_infinity_rather_than_nulling_it(tmp_path):
    """An infinite half-life is the ladder's top rung; written as null it would
    read back as nan and drop out of every sharded median."""
    write_shard(tmp_path, "w", range(0, 1),
                [{"replicate": 0, "up": float("inf"), "down": -np.inf, "hole": np.nan}])
    row = read_shards(tmp_path, "w")[0]
    assert row["up"] == np.inf and row["down"] == -np.inf and np.isnan(row["hole"])
    raw = json.loads(next((tmp_path / "shards").glob("w_*.json")).read_text())
    assert raw["rows"][0]["up"] == "inf"


def test_prediction_8_ignores_a_reference_whose_sweep_selected_nothing():
    sweeps = {"a": {"argmax_k": 2}, "b": {"argmax_k": None}, "c": {"argmax_k": float("nan")},
              "d": {"argmax_k": 8.0}}
    res = e4._prediction_8(sweeps)
    assert res["selected_k"] == [2, 8]
    assert res["max_min_ratio"] == 4.0 and res["supported"]


def test_the_ladder_reports_its_rung_under_the_kernel_it_fitted(corpus):
    """The half-life converters were power-only, so a linear-kernel ladder
    stamped every row with a power-kernel delta_true and covered the wrong
    truth."""
    from lcsa.kernels import LINEAR, delta_from_d_half

    theta0 = fit_constrained(corpus, NAIVE, n_starts=1).theta
    row = e2.fit_rung(corpus, corpus, 4.0, theta0, NAIVE, LINEAR, seed=0, profile=False)
    assert row["delta_true"] == delta_from_d_half(4.0, kernel=LINEAR)
    assert row["delta_true"] != delta_from_d_half(4.0)


def test_e4_takes_the_spillover_lag_by_word_number_when_told_the_positions(corpus):
    """A target the build dropped leaves a hole; lagging over the retained rows
    would hand its successor the surprisal of the word before the hole."""
    rng = np.random.default_rng(5)
    passage = np.array([t.cluster for t in corpus])
    # Word numbers (the targets of a passage are interleaved with the others'),
    # with a hole after the first word of the first passage.
    position = np.array([int((passage[:i] == p).sum()) for i, p in enumerate(passage)])
    position[passage == passage[0]] += (position[passage == passage[0]] >= 1)
    y = 250 + rng.normal(0, 25, len(corpus))
    ctrl = rng.normal(size=(len(corpus), 2))
    s = e4.window_surprisal(corpus, 2)
    by_row = e4.heldout_delta_ll(y, ctrl, s, passage, 2, False, 0)
    by_word = e4.heldout_delta_ll(y, ctrl, s, passage, 2, False, 0, position=position)
    assert np.isfinite(by_row) and np.isfinite(by_word)
    assert by_row != by_word
