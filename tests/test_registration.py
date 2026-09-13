"""The frozen registration: written once, hashed, checked at merge, scored from."""

import json
import math

import pytest

from lcsa.experiments.shards import read_shards, write_shard
from lcsa.registration import (build_registration, check_constants, load_registration,
                               missing_artifacts, registration_hash, score, write_registration)


def test_registration_freezes_once_and_detects_edits(tmp_path):
    reg = build_registration(n_rep=8, n_boot=8, legs=("e6",), frozen_at="2026-09-10T00:00:00+00:00")
    p, h = write_registration(tmp_path, reg)
    assert h == registration_hash(reg) and len(h) == 64
    assert load_registration(tmp_path)[1] == h
    with pytest.raises(FileExistsError, match="frozen once"):
        write_registration(tmp_path, reg)
    edited = json.loads(p.read_text())
    edited["predictions"][0]["support"]["reject_at_most"] = 0.5
    p.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="edited after freezing"):
        load_registration(tmp_path)
    write_registration(tmp_path, reg, force=True)
    assert load_registration(tmp_path)[1] == h
    assert check_constants(reg) == []


def test_code_drift_from_the_frozen_design_is_reported(monkeypatch):
    import lcsa.experiments.e2_ladder as e2

    reg = build_registration(n_rep=8, n_boot=8)
    monkeypatch.setattr(e2, "COVERAGE_RUNGS", (4.0, 8.0))
    drift = check_constants(reg)
    assert drift and "coverage_rungs" in drift[0]


def test_missing_artifacts_names_every_absent_leg(tmp_path):
    gone = missing_artifacts(tmp_path, ["e2", "e6"])
    assert set(gone) == {"e2", "e6"} and gone["e6"] == ["shards/e6_panel_*.json"]


def test_shards_short_of_the_registered_count_are_an_error(tmp_path):
    write_shard(tmp_path, "x", range(0, 4), [{"replicate": b} for b in range(4)])
    assert len(read_shards(tmp_path, "x", n_required=4)) == 4
    with pytest.raises(ValueError, match=r"replicates \[4, 8\) never ran"):
        read_shards(tmp_path, "x", n_required=8)


def _e6_rows(n_rep, shift_share):
    rows = []
    for b in range(n_rep):
        shifted = b < round(shift_share * n_rep)
        for arm, dh in (("plain", 8.0), ("tilted", 32.0 if shifted else 8.5)):
            rows.append({"estimator": "naive", "d_half_true": 8.0, "arm": arm, "replicate": b,
                         "d_half_hat": dh, "covers_truth": True, "unbounded_hi": False})
    return rows


def test_registered_merge_scores_from_the_frozen_file(tmp_path):
    from lcsa.cli import main

    with pytest.raises(FileNotFoundError, match="lcsa register"):
        main(["merge", "--out", str(tmp_path), "--legs", "e6"])
    assert main(["register", "--out", str(tmp_path), "--n-rep", "4", "--legs", "e6"]) == 0
    # Registered, but the leg never ran: the merge names what is missing.
    with pytest.raises(SystemExit, match="missing artifacts.*\n  e6: shards/e6_panel"):
        main(["merge", "--out", str(tmp_path), "--legs", "e6"])
    rows = _e6_rows(4, 0.75)
    write_shard(tmp_path, "e6_panel", range(0, 2), [r for r in rows if r["replicate"] < 2])
    write_shard(tmp_path, "e6_panel", range(2, 4), [r for r in rows if r["replicate"] >= 2])
    assert main(["merge", "--out", str(tmp_path), "--legs", "e6"]) == 0
    card = json.loads((tmp_path / "scorecard.json").read_text())
    by_id = {r["id"]: r for r in card["predictions"]}
    assert by_id[11]["status"] == "supported"
    assert by_id[11]["measured"]["share_shift_over_one_rung"] == 0.75
    assert by_id[3]["status"] == "missing" and card["reading_rule"]["status"] == "missing"
    assert card["registration_sha256"] == load_registration(tmp_path)[1]
    assert "registration_sha256" in (tmp_path / "scorecard.csv").read_text()
    # A registration that asked for more replicates than the shards hold fails the merge.
    main(["register", "--out", str(tmp_path), "--n-rep", "8", "--legs", "e6", "--force"])
    with pytest.raises(ValueError, match="never ran"):
        main(["merge", "--out", str(tmp_path), "--legs", "e6"])
    # A registered leg left out of --legs is refused unless recorded as not run.
    main(["register", "--out", str(tmp_path), "--n-rep", "4", "--legs", "e2,e6", "--force"])
    with pytest.raises(SystemExit, match="registered but not being merged"):
        main(["merge", "--out", str(tmp_path), "--legs", "e6"])
    assert main(["merge", "--out", str(tmp_path), "--legs", "e6", "--allow-missing", "e2"]) == 0
    card = json.loads((tmp_path / "scorecard.json").read_text())
    assert {r["id"]: r["status"] for r in card["predictions"]}[9] == "not run"
    assert main(["merge", "--out", str(tmp_path), "--legs", "e6", "--unregistered"]) == 0
    assert main(["register", "--out", str(tmp_path), "--check"]) == 0


def test_g5_failure_voids_everything_downstream(tmp_path):
    reg = build_registration(n_rep=4, n_boot=4, legs=("e3", "e6"))
    e3 = {"rejection_rates": [
              {"reader": "N0", "estimator": "naive", "reject_cluster_robust": 0.30, "reject_naive_LR": 0.3},
              {"reader": "N0-PRIME", "estimator": "naive", "reject_cluster_robust": 0.05, "reject_naive_LR": 0.6},
              {"reader": "N-TOPIC", "estimator": "naive", "reject_cluster_robust": 0.9, "reject_naive_LR": 0.9}],
          "g5": {"passed": False},
          "human": {"reading_rule": {"verdict": "outside", "floor_check_passed": True,
                                     "human_fraction_debiased": 0.7, "floor_signal_share_median": 0.05,
                                     "thresholds": {"floor_signal_cap": 0.10, "outside_at_least": 0.50,
                                                    "absorbable_at_most": 0.25}}}}
    (tmp_path / "e3_summary.json").write_text(json.dumps(e3))
    card = score(reg, tmp_path)
    by_id = {r["id"]: r for r in card["predictions"]}
    assert by_id[1]["status"] == "falsified" and by_id[1]["measured"]["worst_floor_rate"] == 0.30
    assert by_id[3]["status"] == "void" and by_id[11]["status"] == "missing"
    assert card["reading_rule"]["status"] == "void"
    e3["human"]["reading_rule"]["thresholds"]["outside_at_least"] = 0.4
    (tmp_path / "e3_summary.json").write_text(json.dumps(e3))
    with pytest.raises(ValueError, match="were registered"):
        score(reg, tmp_path)


def _score_12(tmp_path, probe) -> dict:
    reg = build_registration(n_rep=4, n_boot=4, legs=("e1",))
    (tmp_path / "e1_summary.json").write_text(
        json.dumps({"prefix_probe": probe} if probe else {}))
    return {r["id"]: r for r in score(reg, tmp_path)["predictions"]}[12]


def test_prediction_12_reads_the_rows_where_the_two_spans_differ(tmp_path):
    # The overall median is far above the threshold in every case here, so a
    # scorer reading it instead of the truncated median gets the other verdict.
    got = _score_12(tmp_path, {"median_tv_gap": 0.40, "median_tv_gap_truncated": 0.01,
                               "n_truncated_pairs": 40})
    assert got["status"] == "supported" and got["measured"]["median_tv_gap_truncated"] == 0.01
    assert _score_12(tmp_path, {"median_tv_gap": 0.40,
                                "median_tv_gap_truncated": 0.06})["status"] == "falsified"
    assert _score_12(tmp_path, {"median_tv_gap": 0.40,
                                "median_tv_gap_truncated": 0.03})["status"] == "indeterminate"
    absent = _score_12(tmp_path, None)
    assert absent["status"] == "indeterminate"
    assert math.isnan(absent["measured"]["median_tv_gap_truncated"])


def _score_13(tmp_path, delta_capped, uncapped) -> dict:
    reg = build_registration(n_rep=4, n_boot=4, legs=("e3",))
    e3 = {"rejection_rates": [], "g5": {"passed": True},
          "human": {"fits": [{"estimator": "naive", "delta_hat": delta_capped}]}}
    if uncapped is not None:
        e3["human_uncapped"] = uncapped
    (tmp_path / "e3_summary.json").write_text(json.dumps(e3))
    return {r["id"]: r for r in score(reg, tmp_path)["predictions"]}[13]


def test_prediction_13_records_whether_the_uncapped_arm_was_verified(tmp_path):
    """The two arms give the same gap whether the second one was attested
    uncapped or merely deeper, so the scorecard has to carry which it was."""
    fits = [{"estimator": "naive", "delta_hat": 0.30}]
    attested = _score_13(tmp_path, 0.316, {
        "kernel": "power", "fits": fits, "uncapped_verified": True,
        "attestation": "all 16 targets cache their full context, the longest being 13 words"})
    assert attested["status"] == "supported"
    assert attested["measured"]["uncapped_verified"] is True
    assert "full context" in attested["measured"]["uncapped_attestation"]

    older = _score_13(tmp_path, 0.316, {"kernel": "power", "fits": fits})
    assert older["status"] == "supported"
    assert older["measured"]["uncapped_verified"] is False
    assert older["measured"]["uncapped_attestation"] is None


def test_prediction_13_compares_half_lives_not_the_deltas_behind_them(tmp_path):
    got = _score_13(tmp_path, 0.316, {"kernel": "power",
                                      "fits": [{"estimator": "naive", "delta_hat": 0.30}]})
    assert got["status"] == "supported"
    assert got["measured"]["d_half_capped"] == pytest.approx(7.966, abs=1e-3)
    assert got["measured"]["abs_log_half_life_gap"] == pytest.approx(0.1305, abs=1e-3)
    # 0.25 against 0.316 is 0.234 log units apart as deltas and 0.63 as
    # half-lives, so a scorer comparing deltas would call this one supported.
    far = _score_13(tmp_path, 0.316, {"kernel": "power",
                                      "fits": [{"estimator": "naive", "delta_hat": 0.25}]})
    assert far["status"] == "falsified"
    assert far["measured"]["abs_log_half_life_gap"] == pytest.approx(0.6326, abs=1e-3)
    absent = _score_13(tmp_path, 0.316, None)
    assert absent["status"] == "indeterminate"
    assert math.isnan(absent["measured"]["d_half_uncapped"])
    # A fit at the zero-decay boundary has no half-life to compare.
    flat = _score_13(tmp_path, 0.316, {"kernel": "power",
                                       "fits": [{"estimator": "naive", "delta_hat": 0.0}]})
    assert flat["status"] == "indeterminate"
