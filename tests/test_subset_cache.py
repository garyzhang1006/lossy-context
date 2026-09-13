"""The all-subsets cache prediction 10 is scored from, against a stub scorer.

No model is loaded anywhere here.  The scorer below is a deterministic function
of the retained words, so the probabilities are meaningless as physics while
every prompt, bit position, shape and key is the one the real producer computes.
The case that has to hold whatever the weights are is the suffix identity: row
``(1 << k) - 1`` is the depth-``k`` row of the nested cache, and it earns that
name only if it was scored from the very string
:func:`lcsa.cache.context_string` hands the build at depth ``k``.
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pytest

from lcsa.cache import context_string
from lcsa.corpusdata import Corpus
from lcsa.experiments.e1_exactness import (bridging_report, graded_marginal_from_subset,
                                           independent_marginal, subset_masks)
from lcsa.likelihood import NAIVE
from lcsa.store import load_sub_caches, save_sub_caches
from lcsa.subsetcache import (BRIDGING_TARGETS, K_MAX, STEM, mask_context, run,
                              selected_targets, subset_cache, subset_caches)

#: Passages are indexed by Provo word_number, so slot 0 is the word Provo drops
#: from every passage and passage 2 carries a hole in the middle, which is what
#: makes "distance" a count of real words rather than of indices.
PASSAGES = {
    1: ["", "the", "cat", "sat", "on", "the", "mat"],
    2: ["", "a", "very", "", "long", "sentence", "here"],
    3: ["", "some", "words", "before", "the", "target", "word"],
    4: ["", "one", "two", "three", "four", "five", "six"],
    5: ["", "aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj", "kk", "ll"],
}
#: ``targets.csv`` for the fixture: (text_id, word_number) in build order.
KEYS = [(1, 2), (1, 4), (2, 5), (2, 6), (3, 3), (3, 6), (4, 4), (4, 6), (5, 11)]
#: K of each key above, with no depth cap in the way.
DEPTHS = {(1, 2): 1, (1, 4): 3, (2, 5): 3, (2, 6): 4, (3, 3): 2, (3, 6): 5,
          (4, 4): 3, (4, 6): 5, (5, 11): 10}
K_CAP = 3
CANDIDATES = [[f"w{i}{j}" for j in range(4 + i % 3)] for i in range(len(KEYS))]


def _unit(s: str) -> float:
    return zlib.crc32(s.encode("utf-8")) % 2003 / 2003.0 - 0.5


class Provo:
    """The one attribute the producer reads off a :class:`~lcsa.data.provo.ProvoData`."""

    def __init__(self, passages):
        self.passages = dict(passages)


class FakeScorer:
    """A scorer with no model: each candidate is pulled by the words retained.

    The pull is a sum over the retained words, so a deeper context moves the
    distribution further, which is the shape a real nested cache has and which
    gives the bridging fit something other than noise to fit.  Every call is
    recorded, so a test can assert the exact prompt a row was scored under.
    """

    model_name = "fake-deterministic"

    def __init__(self):
        self.contexts = []

    def candidate_logprobs(self, context, words, oov_index=None):
        self.contexts.append(context)
        z = np.array([_unit(f"base|{w}") + 0.6 * sum(_unit(f"{t}|{w}") for t in context.split())
                      for w in words], dtype=np.float64)
        p = np.exp(z - z.max())
        return np.log(p / p.sum())


@pytest.fixture
def provo():
    return Provo(PASSAGES)


@pytest.fixture
def scorer():
    return FakeScorer()


@pytest.fixture
def caches(provo, scorer):
    return subset_caches(provo, KEYS, CANDIDATES, scorer, k_max=K_CAP)


def _corpus_from(caches, keys=None, seed=5):
    """A corpus whose depth rows are the suffix rows of the subset caches.

    Built this way the nested cache and the subset cache are the same numbers on
    the masks they share, which is the premise ``graded_marginal_from_subset``
    rests on and the premise a synthetic corpus is free to establish.
    """
    rng = np.random.default_rng(seed)
    keys = list(caches) if keys is None else list(keys)
    P, n, u, f, g, cl = [], [], [], [], [], []
    for key in keys:
        sub = caches[key]
        K = int(sub.shape[0]).bit_length() - 1
        V = sub.shape[1]
        P.append(sub[[(1 << k) - 1 for k in range(K + 1)]])
        n.append(rng.multinomial(40, np.full(V, 1.0 / V)).astype(float))
        u.append(rng.dirichlet(np.ones(V)))
        f.append(rng.normal(size=(V, 4)))
        g.append((rng.random(V) < 0.3).astype(float))
        cl.append(int(key[0]))
    return Corpus(P, n, u, f, g, cl, feature_names=["a", "b", "c", "d"],
                  target_slots=[0] * len(keys), keys=[list(k) for k in keys])


# -- the bit convention ------------------------------------------------------


def test_the_suffix_masks_are_the_depth_contexts_character_for_character(provo, scorer):
    """Row ``(1 << k) - 1`` is scored under exactly ``context_string(.., k)``."""
    for key in [(1, 4), (2, 5), (3, 3), (4, 4)]:
        tid, wn = key
        K = DEPTHS[key]
        passage = PASSAGES[tid]
        scorer.contexts.clear()
        subset_cache(passage, wn, CANDIDATES[KEYS.index(key)], scorer, K, key=key)
        # One call per mask, in bitmask order, so call m is row m.
        assert len(scorer.contexts) == 1 << K
        for k in range(K + 1):
            want = context_string(passage, wn, k)
            assert mask_context(passage, wn, K, (1 << k) - 1) == want
            assert scorer.contexts[(1 << k) - 1] == want, (key, k)
        assert scorer.contexts[0] == "" and scorer.contexts[(1 << K) - 1] == \
            context_string(passage, wn, K)


def test_a_mask_keeps_the_words_its_bits_name_in_reading_order():
    passage, wn, K = PASSAGES[1], 6, 5
    prefix = [w for w in passage[:wn] if w]  # the cat sat on the
    assert prefix == ["the", "cat", "sat", "on", "the"]
    # Bit 0 is distance 1, the word immediately before the target.
    assert mask_context(passage, wn, K, 0b00001) == "the"
    assert mask_context(passage, wn, K, 0b00010) == "on"
    assert mask_context(passage, wn, K, 0b10000) == "the"
    # A hole in the middle is exactly what the nested cache cannot hold.
    assert mask_context(passage, wn, K, 0b10001) == "the the"
    assert mask_context(passage, wn, K, 0b01101) == "cat sat the"
    assert mask_context(passage, wn, K, 0) == ""


def test_the_depth_cap_drops_everything_beyond_it_whatever_the_mask_says():
    """``K`` short of the real context length is the build's cap, and the masks
    address the nearest ``K`` positions alone."""
    passage, wn = PASSAGES[3], 6  # some words before the target
    assert mask_context(passage, wn, 2, 0b11) == context_string(passage, wn, 2) == "the target"
    assert mask_context(passage, wn, 2, 0b10) == "the"
    assert len(subset_masks(2)) == 4


def test_the_hole_in_a_passage_is_not_a_word(provo):
    """Passage 2 is missing word_number 3, so distance counts skip it."""
    assert context_string(PASSAGES[2], 5, 3) == "a very long"
    assert mask_context(PASSAGES[2], 5, 3, 0b111) == "a very long"
    assert mask_context(PASSAGES[2], 5, 3, 0b100) == "a"


# -- the produced block ------------------------------------------------------


def test_every_block_has_the_shape_and_the_rows_of_a_distribution(caches):
    assert set(caches) == {(1, 2), (1, 4), (2, 5), (3, 3), (4, 4)}
    for key, sub in caches.items():
        K = DEPTHS[key]
        V = len(CANDIDATES[KEYS.index(key)])
        assert sub.shape == (1 << K, V), key
        assert np.all(np.isfinite(sub))
        assert np.allclose(sub.sum(axis=1), 1.0, atol=1e-12), key
        assert np.all(sub > 0.0)


def test_only_the_targets_at_or_below_the_cap_are_taken(provo):
    rows = selected_targets(provo, KEYS, k_max=K_CAP)
    assert [(r["text_id"], r["word_number"]) for r in rows] == [
        (1, 2), (1, 4), (2, 5), (3, 3), (4, 4)]
    assert [r["K"] for r in rows] == [1, 3, 3, 2, 3]
    assert all(r["K"] <= K_CAP for r in rows)
    assert [r["n_contexts_scored"] for r in rows] == [2, 8, 8, 4, 8]
    # The index is a position in targets.csv, which is how the candidate list
    # is found; the artifact keys on the pair rather than on this.
    assert [r["index"] for r in rows] == [KEYS.index(k) for k in
                                          [(1, 2), (1, 4), (2, 5), (3, 3), (4, 4)]]

    # The module default is the cap the registered run uses, and it excludes the
    # ten-deep target whose enumeration would be a thousand contexts.
    assert K_MAX == 8
    wide = {(r["text_id"], r["word_number"]): r["K"] for r in selected_targets(provo, KEYS)}
    assert (5, 11) not in wide and DEPTHS[(5, 11)] == 10
    assert wide == {k: v for k, v in DEPTHS.items() if v <= K_MAX}

    # The build's own depth cap moves K before the selection cap sees it.
    capped = selected_targets(provo, KEYS, k_max=K_CAP, max_depth=2)
    assert len(capped) == len(KEYS) and {r["K"] for r in capped} == {1, 2}


def test_a_target_absent_from_the_provo_record_is_named(scorer):
    with pytest.raises(KeyError, match="passage 99 is absent"):
        selected_targets(Provo(PASSAGES), [(99, 4)])


def test_mismatched_targets_and_candidates_are_refused(provo, scorer):
    with pytest.raises(ValueError, match="candidate lists"):
        subset_caches(provo, KEYS, CANDIDATES[:-1], scorer, k_max=K_CAP)


def test_a_non_finite_row_is_an_error_and_not_a_silent_nan(provo):
    class Broken(FakeScorer):
        def candidate_logprobs(self, context, words, oov_index=None):
            return np.full(len(words), np.inf)

    # inf - inf is the nan the guard catches, and numpy warns on the subtraction
    # before the guard ever sees the row, so the warning is the expected path.
    with np.errstate(invalid="ignore"):
        with pytest.raises(FloatingPointError, match="non-finite subset-cache row"):
            subset_caches(provo, KEYS[:1], CANDIDATES[:1], Broken(), k_max=K_CAP)


# -- the artifact ------------------------------------------------------------


def test_the_artifact_round_trips_and_resolves_to_corpus_indices(caches, tmp_path):
    corpus = _corpus_from(caches)
    p = save_sub_caches(tmp_path / "sub_caches.npz", caches)
    back = load_sub_caches(p, corpus)
    assert sorted(back) == list(range(len(caches)))
    for t, key in enumerate(caches):
        assert tuple(int(x) for x in corpus.keys[t]) == key
        assert back[t].shape == caches[key].shape
        # float32 on disk, renormalised on read exactly as the nested cache is.
        assert np.allclose(back[t], caches[key], atol=1e-6)
        assert np.allclose(back[t].sum(axis=1), 1.0)


def test_the_key_table_and_not_a_position_is_what_resolves(caches, tmp_path):
    """The same file read against a corpus holding the targets in another order
    lands on the same targets, which a stored position would not."""
    p = save_sub_caches(tmp_path / "sub_caches.npz", caches)
    order = list(caches)[::-1]
    corpus = _corpus_from(caches, keys=order)
    back = load_sub_caches(p, corpus)
    for t, key in enumerate(order):
        assert np.allclose(back[t], caches[key], atol=1e-6)


def test_a_stored_target_the_corpus_does_not_hold_is_named(caches, tmp_path):
    p = save_sub_caches(tmp_path / "sub_caches.npz", caches)
    short = _corpus_from(caches, keys=[k for k in caches if k != (2, 5)])
    with pytest.raises(KeyError, match=r"\(2, 5\)"):
        load_sub_caches(p, short)


def test_a_corpus_with_no_keys_says_so_rather_than_guessing(caches, tmp_path):
    p = save_sub_caches(tmp_path / "sub_caches.npz", caches)
    corpus = _corpus_from(caches)
    corpus.keys = None  # a cache written before keys were recorded
    with pytest.raises(ValueError, match="carries no keys"):
        load_sub_caches(p, corpus)


def test_another_npz_is_refused_by_name(caches, tmp_path):
    from lcsa.store import save_corpus

    corpus = _corpus_from(caches)
    p = save_corpus(tmp_path / "cache.npz", corpus)
    with pytest.raises(ValueError, match="artifact kind"):
        load_sub_caches(p, corpus)


def test_the_corpus_keys_survive_the_cache_file(caches, tmp_path):
    from lcsa.store import load_corpus, save_corpus

    corpus = _corpus_from(caches)
    back = load_corpus(save_corpus(tmp_path / "cache.npz", corpus))
    assert back.keys is not None
    assert [tuple(int(x) for x in k) for k in back.keys] == list(caches)
    # with_counts shares everything but the counts, subset_clusters carries the
    # keys of the targets it drew.
    same = back.with_counts([np.ones(back.target(t).V) for t in range(len(back))])
    assert np.array_equal(same.keys, back.keys)
    drawn = back.subset_clusters([0, 0])
    assert drawn.keys is not None and drawn.keys.shape == (2 * sum(
        1 for t in range(len(back)) if back.cluster_index[t] == 0), 2)


def test_run_writes_the_npz_beside_its_selection_record(provo, scorer, tmp_path):
    rec = run(provo, KEYS, CANDIDATES, scorer, tmp_path, k_max=K_CAP)
    assert (tmp_path / f"{STEM}.npz").exists()
    assert (tmp_path / f"{STEM}.json").exists()
    assert (tmp_path / f"{STEM}.csv").exists()
    s = rec["summary"]
    assert s["n_targets_selected"] == 5 and s["n_targets_available"] == len(KEYS)
    assert s["k_max"] == K_CAP and s["max_K_selected"] == 3
    assert s["n_contexts_scored"] == 2 + 8 + 8 + 4 + 8 == len(scorer.contexts)
    assert s["model"] == "fake-deterministic"
    corpus = _corpus_from(subset_caches(provo, KEYS, CANDIDATES, FakeScorer(), k_max=K_CAP))
    assert len(load_sub_caches(tmp_path / f"{STEM}.npz", corpus)) == 5


def test_a_cap_that_selects_nothing_says_what_to_raise(provo, scorer, tmp_path):
    with pytest.raises(ValueError, match="raise --k-max"):
        run(provo, KEYS, CANDIDATES, scorer, tmp_path, k_max=0)


def test_a_selection_that_misses_the_registered_count_is_refused(provo, scorer, tmp_path):
    """The paper prints 439 as a property of the corpus, so a drift must fail here.

    The number reaches five places in the manuscript and a reviewer can check it,
    which it only deserves if a corpus or a depth cap that no longer selects those
    positions stops the producer instead of quietly renaming a different set.
    """
    with pytest.raises(ValueError, match="registration froze 6"):
        run(provo, KEYS, CANDIDATES, scorer, tmp_path, k_max=K_CAP, expect_n=6)
    rec = run(provo, KEYS, CANDIDATES, scorer, tmp_path, k_max=K_CAP, expect_n=5)
    assert rec["summary"]["n_targets_registered"] == 5
    assert run(provo, KEYS, CANDIDATES, scorer, tmp_path,
               k_max=K_CAP)["summary"]["n_targets_registered"] is None


def test_the_registration_freezes_the_bridging_constants():
    """``lcsa merge`` refuses a code constant that drifted, and 439 is now one.

    ``BRIDGING_TARGETS`` is the count the Provo selection has to return, and the
    registration is the only place the paper's number and the producer's cap meet,
    so a change to either has to show up as drift rather than as a quiet edit.
    """
    from lcsa.registration import build_registration, check_constants

    reg = build_registration()
    assert reg["design"]["bridging_targets"] == BRIDGING_TARGETS == 439
    assert reg["design"]["bridging_k_max"] == K_MAX == 8
    assert check_constants(reg) == []
    drifted = json.loads(json.dumps(reg))
    drifted["design"]["bridging_targets"] = 440
    assert any("bridging_targets" in m for m in check_constants(drifted))
    stale = json.loads(json.dumps(reg))
    del stale["design"]["bridging_targets"]
    assert any("predates this constant" in m for m in check_constants(stale))


# -- the leg that reads it ---------------------------------------------------


def test_bridging_runs_end_to_end_on_the_produced_caches(caches):
    corpus = _corpus_from(caches)
    subs = {t: caches[key] for t, key in enumerate(caches)}
    rec = bridging_report(corpus, subs, NAIVE, d_half_true=8.0, n_boot=5, seed=3)
    assert rec["n_targets"] == 5 and rec["n_clusters"] == 4
    for k in ("mean_kl_indep_given_graded", "median_kl_indep_given_graded",
              "mean_tv_between_kernels", "delta_hat_graded", "delta_true"):
        assert np.isfinite(rec[k]), k
    assert rec["mean_kl_indep_given_graded"] >= 0.0
    assert 0.0 <= rec["mean_tv_between_kernels"] <= 1.0
    assert rec["n_boot"] == 5 and isinstance(rec["agrees_within_0.25_log_units"], bool)


def test_the_two_kernels_read_the_same_block_on_their_own_masks(caches):
    """Graded truncation reads the suffix rows and independent deletion reads
    all of them, and both are proper distributions over the same candidates."""
    key = (1, 4)
    sub = caches[key]
    K = DEPTHS[key]
    for delta in (0.0, 0.316, 2.0):
        ind = independent_marginal(sub, K, delta)
        grd = graded_marginal_from_subset(sub, K, delta)
        assert np.isclose(ind.sum(), 1.0) and np.isclose(grd.sum(), 1.0)
        assert np.all(ind > 0) and np.all(grd > 0)
    # At delta = 0 both keep everything, so both are the full-context row.
    assert np.allclose(independent_marginal(sub, K, 0.0), sub[(1 << K) - 1])
    assert np.allclose(graded_marginal_from_subset(sub, K, 0.0), sub[(1 << K) - 1])


def test_the_e1_leg_folds_bridging_in_only_when_it_is_given_caches(caches, tmp_path):
    import json

    from lcsa.experiments.e1_exactness import run as run_e1

    corpus = _corpus_from(caches)
    subs = {t: caches[key] for t, key in enumerate(caches)}
    assert "bridging" not in run_e1(corpus, [NAIVE], tmp_path / "without")
    res = run_e1(corpus, [NAIVE], tmp_path / "with", sub_caches=subs, seed=3)
    assert res["bridging"]["n_targets"] == 5
    on_disk = json.loads((tmp_path / "with" / "e1_summary.json").read_text())
    assert on_disk["bridging"]["n_targets"] == 5


# -- the command line --------------------------------------------------------


def test_the_cli_exposes_the_producer_with_the_module_defaults():
    from lcsa.cli import build_parser, cmd_sub_cache

    args = build_parser().parse_args(["sub-cache", "--provo-dir", "somewhere"])
    assert args.func is cmd_sub_cache
    assert args.k_max == K_MAX == 8
    assert args.max_depth is None
    assert args.targets.endswith("targets.csv") and args.candidates.endswith("candidates.json")
    assert args.out.endswith("build"), "the artifact belongs beside the targets it keys on"
    help_text = build_parser()._subparsers._group_actions[0].choices["sub-cache"].format_help()
    assert "GPU" in build_parser().format_help() and help_text


def test_the_e1_leg_takes_the_artifact_from_the_command_line(caches, tmp_path):
    from lcsa.cli import build_parser, main
    from lcsa.store import save_corpus

    assert build_parser().parse_args(["e1"]).sub_caches is None
    assert build_parser().parse_args(["e1", "--sub-caches", "s.npz"]).sub_caches == "s.npz"
    # A real cache has to be on disk first, because cmd_e1 loads it before it
    # ever looks at --sub-caches, and the build hint would otherwise be raised.
    cache = save_corpus(tmp_path / "cache.npz", _corpus_from(caches))
    with pytest.raises(SystemExit, match="Run .lcsa sub-cache. first"):
        main(["e1", "--cache", str(cache), "--out", str(tmp_path / "art"),
              "--sub-caches", str(tmp_path / "missing.npz")])


def test_the_sbatch_job_and_the_pipeline_put_it_between_the_build_and_e1():
    from pathlib import Path

    slurm = Path(__file__).resolve().parents[1] / "slurm"
    job = (slurm / "sub_cache.sbatch").read_text()
    assert "lcsa --verbose sub-cache" in job
    assert "cuda.is_available" in job and "scu-gpu" in job
    pipeline = (slurm / "pipeline.sh").read_text().splitlines()
    sub = next(ln for ln in pipeline if ln.startswith("SUB="))
    e1 = next(ln for ln in pipeline if ln.startswith("E1="))
    assert "afterok:$BUILD" in sub and "sub_cache.sbatch" in sub
    assert "afterok:$SUB" in e1
    assert "--sub-caches" in (slurm / "e1.sbatch").read_text()
