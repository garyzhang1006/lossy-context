import json

import numpy as np

from lcsa.experiments import e3_nulls as e3
from lcsa.experiments import e6_crossed as e6
from lcsa.kernels import POWER
from lcsa.likelihood import NAIVE, REPAIRED
from conftest import draw_true_delta, make_corpus


def _prepared(tmp_path):
    corpus = draw_true_delta(make_corpus(n_targets=40, n_clusters=5, seed=61), 0.316, seed=7)
    # Any direction list under the N-ORDER name gives the panel its tilt; the
    # lexical direction stands in for the order one that needs Provo text.
    h = {"N-ORDER": e3.h_lexical(corpus, REPAIRED, seed=0)}
    prep = e3.prepare(corpus, [NAIVE], tmp_path / "prep", h_specs=h, seed=0,
                      readers=["N0", "N-ORDER"])
    return corpus, prep


def test_panel_shards_reproduce_the_monolithic_run(tmp_path):
    corpus, prep = _prepared(tmp_path)
    theta0 = np.asarray(prep["theta0"], dtype=np.float64)
    mono, sh = tmp_path / "mono", tmp_path / "sh"
    res = e6.run(corpus, theta0, prep, [NAIVE], mono, POWER, n_rep=4, seed=0, rungs=(8.0,))
    e6.run_shard(corpus, theta0, prep, [NAIVE], sh, range(0, 2), POWER, 0, (8.0,))
    e6.run_shard(corpus, theta0, prep, [NAIVE], sh, range(2, 4), POWER, 0, (8.0,))
    merged = e6.merge(sh)
    assert (mono / "e6_panel.csv").read_bytes() == (sh / "e6_panel.csv").read_bytes()
    assert merged["panel"] == json.loads((mono / "e6_summary.json").read_text())["panel"]
    row = res["panel"][0]
    assert row["estimator"] == "naive" and row["d_half_true"] == 8.0
    assert row["n_pairs"] == 4
    assert np.isfinite(row["median_d_half_plain"])


def test_tilted_arm_shares_its_retention_draws_with_the_plain_arm(tmp_path):
    corpus, prep = _prepared(tmp_path)
    theta0 = np.asarray(prep["theta0"], dtype=np.float64)
    rows = e6.panel_replicates(corpus, theta0, prep, NAIVE, reps=range(1), rungs=(8.0,),
                               seed=0, profile=False)
    assert [r["arm"] for r in rows] == ["plain", "tilted"]
    assert not any(r["failed"] for r in rows)
    assert rows[0]["alpha"] == 0.0 and rows[1]["alpha"] > 0.0
    # Same seed and rung: the retention draws match, so the two arms differ
    # only through the tilt and the summary's paired shift is meaningful.
    assert rows[0]["replicate"] == rows[1]["replicate"] == 0
