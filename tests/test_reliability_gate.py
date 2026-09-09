"""G3 from the cache and the eye-tracking arm, and the collected gate table."""

import json

import numpy as np

from lcsa.reliability import split_half_reliability
from test_build_pipeline import built, provo_dir, scorer  # noqa: F401


def test_participant_halves_split_by_reader_not_by_row():
    rng = np.random.default_rng(0)
    items = np.repeat(np.arange(30), 8)
    parts = np.tile(np.arange(8), 30)
    true = rng.normal(size=30)[items]
    values = true + 0.3 * rng.normal(size=true.size)
    r_part = split_half_reliability(values, items, participants=parts, n_splits=40)
    r_row = split_half_reliability(values, items, n_splits=40)
    assert 0.8 < r_part <= 1.0 and 0.8 < r_row <= 1.0
    assert np.isnan(split_half_reliability(values, items, participants=np.zeros(values.size)))


def test_reliability_stage_writes_g3(built, tmp_path):
    from lcsa.experiments.reliability_stage import run

    provo, corpus, _, _ = built
    rec = run(corpus, provo, tmp_path, n_splits=4, n_sim=4)
    assert rec["n_participants"] == 4
    assert np.isfinite(rec["gaze_split_half"])
    saved = json.loads((tmp_path / "reliability.json").read_text())
    assert saved["g3"]["name"] == "G3" and isinstance(saved["g3"]["passed"], bool)
    assert "js_split_half_debiased" in saved["js"]


def test_gates_table_reports_missing_gates_as_not_run(tmp_path):
    from lcsa.experiments.gates_table import collect

    build = tmp_path / "build"
    build.mkdir()
    (build / "g1.json").write_text(json.dumps({"tflops": 3.1, "passed": True,
                                               "tokens_forwarded": 10}))
    (tmp_path / "e3_summary.json").write_text(json.dumps({
        "g5": {"name": "G5", "passed": True, "measured": {"N0": 0.04}, "threshold": "<= 0.10",
               "fallback": "", "notes": ""}, "human": None}))
    rows = collect(tmp_path, build, gpu_hours=11.0, e3_complete=False)
    by = {r["gate"]: r for r in rows}
    assert [r["gate"] for r in rows] == [f"G{i}" for i in range(8)]
    assert by["G1"]["passed"] is True and by["G1"]["m_tflops"] == 3.1
    assert by["G5"]["passed"] is True and by["G5"]["m_N0"] == 0.04
    assert by["G0"]["passed"] is None and by["G0"]["note"] == "not run"
    assert by["G7"]["passed"] is False
    assert (tmp_path / "gates.csv").exists()


def test_reliability_and_gates_cli(built, provo_dir, tmp_path):
    from lcsa.cli import main
    from lcsa.store import save_corpus

    _, corpus, _, _ = built
    cache = tmp_path / "cache.npz"
    save_corpus(cache, corpus)
    out = tmp_path / "art"
    assert main(["reliability", "--cache", str(cache), "--out", str(out),
                 "--provo-dir", str(provo_dir), "--n-splits", "3", "--n-sim", "3"]) == 0
    assert main(["gates", "--out", str(out)]) == 0
    rows = json.loads((out / "gates.json").read_text())
    assert {r["gate"]: r["passed"] for r in rows}["G3"] is not None
