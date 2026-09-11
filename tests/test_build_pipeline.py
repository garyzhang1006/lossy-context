"""Provo plus a reference model to a fitted corpus, with nothing mocked in between.

This covers the one seam the synthetic tests cannot reach: the candidate set,
the nested cache, the target key file, and the alignment of the eye-tracking arm
to the built targets.  The reference model is a two-layer transformer with random
weights, so the numbers are meaningless while every shape, index and join is real.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_provo_loader import PASSAGES, _eye_rows, _norms_rows
from test_tokenization import ByteTokenizer

from lcsa.build import BuildConfig, build_corpus, gate_g0
from lcsa.data.provo import canonical_word, load_provo
from lcsa.data.subtlex import load_subtlex
from lcsa.experiments.e4_reading import gaze_table, window_surprisal
from lcsa.fitting import fit
from lcsa.likelihood import NAIVE, loglik
from lcsa.store import load_corpus, save_corpus
from lcsa.tokenization import OOV

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def provo_dir(tmp_path_factory):
    """Module-scoped copy of the loader fixture; the build below is too slow to repeat."""
    d = tmp_path_factory.mktemp("provo")
    norms = _norms_rows()
    norms.loc[0, "Word"] = "Th\u00e9"
    norms.to_csv(d / "Provo_Corpus-Predictability_Norms.csv", index=False,
                 encoding="latin-1")
    _eye_rows().to_csv(d / "Provo_Corpus-Eyetracking_Data.csv", index=False)
    return d


@pytest.fixture(scope="module")
def scorer():
    from transformers import GPT2Config, GPT2LMHeadModel

    from lcsa.cache import ReferenceScorer

    torch.manual_seed(3)
    cfg = GPT2Config(vocab_size=256, n_positions=256, n_embd=32, n_layer=2, n_head=2)
    return ReferenceScorer("test-tiny", device="cpu", dtype="float32",
                           model=GPT2LMHeadModel(cfg), tokenizer=ByteTokenizer())


@pytest.fixture(scope="module")
def built(provo_dir, scorer):
    provo = load_provo(provo_dir, require_eye=True)
    cfg = BuildConfig(max_candidates=20, max_depth=4, top_k_expansions=0)
    words, keys = [], []
    corpus = build_corpus(provo, scorer, load_subtlex(None), cfg,
                          keep_words=words, keep_keys=keys)
    return provo, corpus, words, keys


def test_every_non_initial_target_is_built(built):
    """Word 1 of a passage has no context, so it carries no cache and no target."""
    provo, corpus, _, keys = built
    expected = sum(len(w) - 1 for w in PASSAGES.values())
    assert len(corpus) == expected
    assert all(wn > 1 for _, wn in keys)


def test_keys_are_returned_in_corpus_order(built):
    provo, corpus, words, keys = built
    assert len(keys) == len(corpus) == len(words)
    for (tid, _), tgt in zip(keys, corpus):
        assert tgt.cluster == list(sorted({k[0] for k in keys})).index(tid)


def test_depth_is_the_number_of_preceding_words_capped_at_the_config(built):
    provo, corpus, _, keys = built
    for (tid, wn), tgt in zip(keys, corpus):
        # Passages are indexed by word_number and Provo numbers from 2, so the
        # words below the target are numbered 2..wn-1, which is wn - 2 of them.
        assert tgt.K == min(wn - 2, 4)


def test_cache_rows_are_distributions_and_differ_across_depth(built):
    _, corpus, _, _ = built
    for tgt in corpus:
        assert np.allclose(tgt.P.sum(axis=1), 1.0, atol=1e-6)
        if tgt.K > 0:
            assert np.abs(tgt.P[-1] - tgt.P[0]).max() > 1e-9


def test_gate_g0_passes_on_the_built_cache(built):
    provo, corpus, _, _ = built
    g = gate_g0(corpus, provo)
    assert g["passed"], g
    assert g["rows_not_normalised"] == 0
    assert g["degenerate_targets"] == 0


def test_responses_are_conserved_by_the_candidate_set(built):
    """Every response either lands on a candidate or in the bucket; none is lost."""
    provo, corpus, words, keys = built
    for (tid, wn), tgt, ws in zip(keys, corpus, words):
        grp = provo.responses[(provo.responses["text_id"] == tid)
                              & (provo.responses["word_number"] == wn)]
        assert tgt.N == pytest.approx(float(grp["count"].sum()))
        assert OOV in ws


def test_the_corpus_word_is_in_its_own_candidate_set(built):
    provo, corpus, words, keys = built
    wmap = {(int(r.text_id), int(r.word_number)): str(r.word)
            for r in provo.words.itertuples()}
    for (tid, wn), tgt, ws in zip(keys, corpus, words):
        assert tgt.target_slot >= 0
        assert ws[tgt.target_slot] == canonical_word(wmap[(tid, wn)])


def test_prior_mention_flags_words_already_seen_in_the_passage(built):
    provo, corpus, words, keys = built
    for (tid, wn), tgt, ws in zip(keys, corpus, words):
        seen = {canonical_word(x) for x in provo.passages[tid][:wn] if x}
        assert [bool(x) for x in tgt.g] == [w in seen for w in ws]


def test_features_have_the_documented_columns(built):
    _, corpus, _, _ = built
    assert corpus.M == 4
    assert corpus.feature_names == ["log_unigram", "length", "is_content", "log_docfreq"]
    assert np.all(np.isfinite(corpus.target(0).f))


def test_built_corpus_survives_a_disk_round_trip_and_a_fit(built, tmp_path):
    _, corpus, _, _ = built
    p = tmp_path / "cache.npz"
    save_corpus(p, corpus)
    back = load_corpus(p)
    th = np.array([0.3, 0.1, 0.9])
    assert loglik(back, th, NAIVE) == pytest.approx(loglik(corpus, th, NAIVE), rel=1e-4)
    f = fit(back, NAIVE, n_starts=2, seed=0)
    assert np.isfinite(f.loglik) and f.delta >= 0.0


def test_gaze_table_aligns_to_the_built_targets_and_never_imputes(built):
    provo, corpus, _, keys = built
    y, ctrl, passage = gaze_table(provo, keys, load_subtlex(None))
    assert y.shape == (len(corpus),)
    assert ctrl.shape == (len(corpus), 2)
    assert list(passage) == [t for t, _ in keys]
    # Passage 2 word_numbers 3 and 7 have no eye-tracking record in the fixture.
    missing = {(t, w) for (t, w), v in zip(keys, y) if not np.isfinite(v)}
    assert missing == {(2, 3), (2, 7)}


def test_window_surprisal_is_finite_where_the_target_word_is_a_candidate(built):
    _, corpus, _, _ = built
    s = window_surprisal(corpus, 2)
    assert s.shape == (len(corpus),)
    assert np.all(np.isfinite(s))
    assert np.all(s > 0)


def test_a_selection_predicate_restricts_the_build(provo_dir, scorer):
    provo = load_provo(provo_dir)
    keys = []
    sub = build_corpus(provo, scorer, load_subtlex(None),
                       BuildConfig(max_candidates=20, max_depth=4, top_k_expansions=0),
                       select=lambda tid, wn, K: K <= 2, keep_keys=keys)
    assert len(sub) == len(keys) > 0
    assert all(t.K <= 2 for t in sub)


def test_an_empty_build_says_what_to_check(provo_dir, scorer):
    provo = load_provo(provo_dir)
    with pytest.raises(ValueError, match="no target survived"):
        build_corpus(provo, scorer, load_subtlex(None),
                     BuildConfig(top_k_expansions=0),
                     select=lambda tid, wn, K: False)


def test_frozen_candidates_reproduce_the_candidate_sets_under_another_scorer(built, provo_dir):
    """A reference cache must have row for row the primary's candidates."""
    from transformers import GPT2Config, GPT2LMHeadModel

    from lcsa.cache import ReferenceScorer

    provo, corpus, words, keys = built
    torch.manual_seed(11)
    cfg = GPT2Config(vocab_size=256, n_positions=256, n_embd=32, n_layer=2, n_head=2)
    other = ReferenceScorer("test-other", device="cpu", dtype="float32",
                            model=GPT2LMHeadModel(cfg), tokenizer=ByteTokenizer())
    w2, k2 = [], []
    ref = build_corpus(provo, other, load_subtlex(None),
                       BuildConfig(max_candidates=20, max_depth=4, top_k_expansions=0),
                       keep_words=w2, keep_keys=k2, candidates=dict(zip(keys, words)))
    assert k2 == keys and w2 == words
    assert [t.V for t in ref] == [t.V for t in corpus]
    assert [t.target_slot for t in ref] == [t.target_slot for t in corpus]
    assert all(np.allclose(a.n, b.n) for a, b in zip(ref, corpus))
    assert not all(np.allclose(a.P, b.P) for a, b in zip(ref, corpus))
    # Every target the frozen set lacks is skipped rather than rebuilt freely.
    partial = build_corpus(provo, other, load_subtlex(None),
                           BuildConfig(max_candidates=20, max_depth=4, top_k_expansions=0),
                           candidates=dict(zip(keys[:3], words[:3])))
    assert len(partial) == 3


def test_e4_cli_stages_accept_a_frozen_reference_build(built, provo_dir, tmp_path):
    from lcsa.cli import _targets_csv, main

    provo, corpus, words, keys = built
    prim, ref = tmp_path / "build", tmp_path / "ref"
    for d in (prim, ref):
        d.mkdir()
        save_corpus(d / "cache.npz", corpus)
        _targets_csv(d / "targets.csv", keys)
    (prim / "candidates.json").write_text(json.dumps(words))
    out = tmp_path / "art"
    args = ["e4", "--cache", str(prim / "cache.npz"), "--out", str(out), "--provo-dir",
            str(provo_dir), "--targets", str(prim / "targets.csv"), "--estimators", "naive",
            "--reference", f"self={ref}", "--n-folds", "3"]
    assert main([*args, "--stage", "sweep"]) == 0
    assert main([*args, "--stage", "boot", "--n-boot", "3", "--boot-stop", "2"]) == 0
    assert main([*args, "--stage", "boot", "--n-boot", "3", "--boot-start", "2"]) == 0
    assert main(["register", "--out", str(out), "--n-boot", "3", "--legs", "e4"]) == 0
    assert main(["merge", "--out", str(out), "--legs", "e4", "--estimators", "naive"]) == 0
    assert (out / "scorecard.csv").exists()
    res = json.loads((out / "e4_summary.json").read_text())
    assert res["selected"]["self"] == res["selected"]["primary"]
    assert res["argmax_bootstrap"]["self"]["n_boot"] == 3
    _targets_csv(ref / "targets.csv", keys[1:] + keys[:1])
    with pytest.raises(SystemExit, match="different targets"):
        main([*args, "--stage", "sweep"])


def test_passage_perplexity_is_finite_and_counts_every_token(built, scorer):
    from lcsa.build import provo_perplexity

    provo = built[0]
    rec = provo_perplexity(provo, scorer)
    # The empty slots are word numbers Provo does not carry, so they are dropped
    # before the join exactly as provo_perplexity drops them before scoring.
    n_expected = sum(len(" ".join(w for w in ws if w).encode("utf-8"))
                     for ws in provo.passages.values())
    assert rec["n_tokens"] == n_expected
    assert np.isfinite(rec["perplexity"]) and rec["perplexity"] > 1.0
    assert set(rec["per_passage"]) == set(provo.passages)
    assert set(rec["per_passage_min_k"]) == set(provo.passages)
    # Min-K% averages the least likely fifth, so it sits at or below the mean.
    for tid, ws in provo.passages.items():
        nll, n, tokens = scorer.passage_nll([w for w in ws if w])
        assert tokens.shape == (n,) and abs(tokens.sum() - nll) < 1e-6
        assert rec["per_passage_min_k"][tid] <= -nll / n + 1e-9


def test_build_writes_g1_from_tokens_forwarded(provo_dir, scorer, tmp_path, monkeypatch):
    import lcsa.cache
    from lcsa.cli import main

    # The CLI constructs the scorer from a checkpoint name; hand it the fixture.
    monkeypatch.setattr(lcsa.cache, "ReferenceScorer", lambda *a, **k: scorer)
    out = tmp_path / "b"
    assert main(["build", "--provo-dir", str(provo_dir), "--model", "test-tiny",
                 "--out", str(out), "--max-depth", "3", "--max-candidates", "12",
                 "--top-k", "0"]) == 0
    g1 = json.loads((out / "g1.json").read_text())
    assert g1["tokens_forwarded"] > 0 and g1["n_params"] > 0
    assert np.isfinite(g1["tflops"]) and g1["passed"] is False
    ppl = json.loads((out / "perplexity.json").read_text())
    assert ppl["min_k_frac"] == 0.2 and len(ppl["per_passage_min_k"]) == 2


def test_confounds_command_prints_delta_beside_perplexity(built, tmp_path):
    from lcsa.cli import main

    provo, corpus, words, keys = built
    d = tmp_path / "ref"
    d.mkdir()
    save_corpus(d / "cache.npz", corpus)
    (d / "perplexity.json").write_text(json.dumps({"perplexity": 12.5, "n_tokens": 100}))
    out = tmp_path / "art"
    assert main(["confounds", "--reference", f"gpt2={d}", "--cache", str(d / "cache.npz"),
                 "--out", str(out), "--estimators", "naive"]) == 0
    rows = json.loads((out / "confounds.json").read_text())["rows"]
    assert rows[0]["reader"] == "gpt2" and rows[0]["perplexity"] == 12.5
    assert rows[0]["kind"] == "competence confound"
    assert np.isfinite(rows[0]["delta_hat"])
    # Two passages cannot form three tertiles of two, so no tertile rows appear.
    assert all(r["kind"] == "competence confound" for r in rows)


def test_min_k_tertiles_refit_the_human_counts_by_passage(corpus, tmp_path):
    from lcsa.experiments.confounds import min_k_tertiles, run
    from lcsa.likelihood import NAIVE

    ppl = {"perplexity": 20.0, "n_tokens": 500,
           "per_passage_min_k": {str(int(c)): -float(i) for i, c in enumerate(corpus.cluster_ids)}}
    tert = min_k_tertiles(corpus, ppl)
    assert [t for t, *_ in tert] == [1, 2, 3]
    assert sum(len(idx) for _, idx, _, _ in tert) == corpus.n_clusters
    assert tert[0][2] <= tert[0][3] <= tert[1][2]
    rows = run({}, [NAIVE], tmp_path, primary=(corpus, ppl))
    assert [r["tertile"] for r in rows] == [1, 2, 3]
    assert all(r["kind"] == "min-k tertile" and np.isfinite(r["delta_hat"]) for r in rows)
    assert sum(r["n_clusters"] for r in rows) == corpus.n_clusters
    assert min_k_tertiles(corpus, {"per_passage_min_k": {}}) == []
