"""The two arms of E3: the paired contrast and the restart budgets.

Both defects here were silent by construction.  The contrast reconstructed its
bootstrap draw from the resampled corpus's own cluster labels, which
``subset_clusters`` has already relabelled, so the null arm was refitted on the
whole corpus every replicate and the only statistic that could have caught it,
the pairing correlation, was a correlation against a constant.  The restart
budgets differed between the arm fitted on the human counts and the arm fitted
on the replicates, so the likelihoods behind the rejection-rate table were
searched to different depths.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import draw_true_delta, make_corpus

from lcsa.experiments import e3_nulls as e3
from lcsa.experiments.e3_nulls import (FIT_STARTS, SCORE_STARTS, contrast_replicates,
                                       fit_and_profile, summarise_contrast)
from lcsa.likelihood import NAIVE


def test_subset_clusters_records_the_draw_its_relabelling_destroys():
    corpus = make_corpus(n_targets=40, n_clusters=8, seed=200)
    draw = [3, 3, 0, 5, 5, 5, 1, 7]
    sub = corpus.subset_clusters(draw)
    assert sub.source_clusters.tolist() == draw
    # The reason the draw has to be recorded: the labels carry none of it.
    assert np.unique(sub.cluster_index).tolist() == list(range(len(draw)))
    assert corpus.source_clusters.tolist() == list(range(corpus.n_clusters))


def test_the_null_arm_is_refitted_on_the_passages_the_replicate_drew():
    """A null arm refitted on the full corpus is the same number every replicate,
    which leaves the contrast unpaired and its pairing correlation undefined."""
    base = make_corpus(n_targets=60, n_clusters=10, seed=201)
    human = draw_true_delta(base, 0.5, seed=202, n_per_target=60)
    null = draw_true_delta(base, 0.2, seed=203, n_per_target=60)

    reps = contrast_replicates(human, {"N-FAKE": null}, NAIVE, reps=range(8), seed=0)
    nulls = np.array([r["N-FAKE"] for r in reps], dtype=np.float64)
    humans = np.array([r["human"] for r in reps], dtype=np.float64)
    assert np.isfinite(nulls).all() and np.isfinite(humans).all()
    assert nulls.std() > 0.0, "the null arm did not move with the resample"

    summary = summarise_contrast(reps, ["N-FAKE"])["contrasts"]["N-FAKE"]
    assert summary["n_usable"] > 3
    assert np.isfinite(summary["pairing_correlation"])
    assert np.isfinite(summary["se_log_diff"]) and summary["se_log_diff"] > 0.0


def _budget_spy(monkeypatch):
    """Record the multistart budget of every top-level fit and score test."""
    seen: dict[str, list[int]] = {"fit": [], "score_test": []}
    real_fit, real_score = e3.fit, e3.score_test

    def fit_spy(*args, n_starts=3, **kw):
        seen["fit"].append(int(n_starts))
        return real_fit(*args, n_starts=n_starts, **kw)

    def score_spy(*args, n_starts=3, **kw):
        seen["score_test"].append(int(n_starts))
        return real_score(*args, n_starts=n_starts, **kw)

    monkeypatch.setattr(e3, "fit", fit_spy)
    monkeypatch.setattr(e3, "score_test", score_spy)
    return seen


def test_both_arms_of_the_rate_table_are_searched_to_the_same_depth(monkeypatch):
    """A deeper multistart can only raise the attained likelihood, so an arm
    searched harder carries that advantage straight into its rejection rate."""
    corpus = draw_true_delta(make_corpus(n_targets=40, n_clusters=8, seed=204), 0.316,
                             seed=205, n_per_target=40)
    seen = _budget_spy(monkeypatch)

    fit_and_profile(corpus, NAIVE, seed=0, profile=False)
    human = {k: list(v) for k, v in seen.items()}
    seen["fit"].clear()
    seen["score_test"].clear()

    row = e3._replicate_fit(corpus, NAIVE, e3.POWER, 0)
    assert not row["failed"], row.get("error")

    assert human == {k: list(v) for k, v in seen.items()}
    assert human == {"fit": [FIT_STARTS], "score_test": [SCORE_STARTS]}


def _capped(corpus, cap):
    """The same counts on a depth ladder cut at ``cap``, as ``--max-depth`` cuts it."""
    from lcsa.corpusdata import Corpus

    tg = [corpus.target(t) for t in range(corpus.n_targets)]
    return Corpus([t.P[:cap + 1] for t in tg], [t.n for t in tg], [t.u for t in tg],
                  [t.f for t in tg], [t.g for t in tg], list(corpus.clusters),
                  [t.word_ids for t in tg], corpus.feature_names,
                  list(corpus.target_slots))


def test_the_uncapped_arm_is_recorded_beside_the_capped_human_fit(tmp_path):
    """Prediction 13 refits the human counts on a cache whose depths were not capped."""
    from lcsa.kernels import d_half_from_delta

    full = draw_true_delta(make_corpus(n_targets=24, n_clusters=6, seed=210), 0.316,
                           seed=211, n_per_target=60)
    capped = _capped(full, 6)
    res = e3.run(capped, [NAIVE], tmp_path, n_rep=2, n_boot=2, seed=0, readers=["N0"],
                 uncapped=full)

    arm = res["human_uncapped"]
    assert arm["kernel"] == "power"
    assert arm["mean_K"] == full.mean_K > arm["mean_K_capped"] == capped.mean_K
    assert [f["estimator"] for f in arm["fits"]] == ["naive"]
    assert np.isfinite(d_half_from_delta(arm["fits"][0]["delta_hat"]))
    on_disk = json.loads((tmp_path / "e3_summary.json").read_text())
    assert on_disk["human_uncapped"]["fits"][0]["delta_hat"] == arm["fits"][0]["delta_hat"]
    assert json.loads((tmp_path / "e3_human_stage.json").read_text())["uncapped"] == arm


def _attested(corpus, n_context, max_depth):
    """The same rows carrying the build record ``lcsa build`` now writes."""
    from lcsa.corpusdata import Corpus

    tg = [corpus.target(t) for t in range(corpus.n_targets)]
    return Corpus([t.P for t in tg], [t.n for t in tg], [t.u for t in tg],
                  [t.f for t in tg], [t.g for t in tg], list(corpus.clusters),
                  [t.word_ids for t in tg], corpus.feature_names,
                  list(corpus.target_slots), n_context=n_context, max_depth=max_depth)


def test_the_uncapped_arm_says_whether_the_build_attested_it(tmp_path):
    """A deeper cache is not an uncapped one, and only the build record tells
    them apart; a cache written before that record degrades to the old check."""
    full = draw_true_delta(make_corpus(n_targets=16, n_clusters=4, seed=214), 0.316,
                           seed=215, n_per_target=40)
    capped = _capped(full, 6)
    depths = [full.target(t).K for t in range(full.n_targets)]

    old = e3.uncapped_human_fits(full, capped, [NAIVE])
    assert old["uncapped_verified"] is False and old["max_depth"] is None
    assert "records no per-target context length" in old["attestation"]

    arm = e3.uncapped_human_fits(_attested(full, depths, 64), capped, [NAIVE])
    assert arm["uncapped_verified"] is True and arm["max_depth"] == 64
    assert str(max(depths)) in arm["attestation"]

    # Deeper than the capped arm and still truncated: exactly the case the mean
    # depth comparison passes and the prediction's statement does not describe.
    still_cut = _attested(_capped(full, 5), [d + 3 for d in depths], 5)
    with pytest.raises(ValueError, match="still truncated at max_depth 5"):
        e3.uncapped_human_fits(still_cut, _capped(full, 4), [NAIVE])


def test_an_uncapped_cache_that_is_not_deeper_is_refused(tmp_path):
    full = draw_true_delta(make_corpus(n_targets=16, n_clusters=4, seed=212), 0.316,
                           seed=213, n_per_target=40)
    with pytest.raises(ValueError, match="no deeper than the capped"):
        e3.uncapped_human_fits(_capped(full, 6), full, [NAIVE])
    with pytest.raises(ValueError, match="same targets rebuilt"):
        e3.uncapped_human_fits(full, _capped(full, 6).subset_clusters([0, 1]), [NAIVE])
