"""The prefix-restoration probe: what the truncated prompt costs, bounded above.

The reference model here is a two-layer transformer with random weights, so the
gaps are meaningless as physics while every prompt, index and aggregation is the
one the real probe computes.  The cases that must hold whatever the weights are
the degenerate ones: a depth whose two prompts are the same string has to score
zero, not nearly zero.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_provo_loader import FIRST_WORD_NUMBER, PASSAGES, _eye_rows, _norms_rows
from test_tokenization import ByteTokenizer

from lcsa.build import BuildConfig, build_corpus
from lcsa.data.provo import load_provo
from lcsa.data.subtlex import load_subtlex
from lcsa.manifest import write_manifest
from lcsa.prefixprobe import (DEFAULT_SAMPLE, DEFAULT_SEED, DIAGNOSTIC,
                              prefix_restoration_probe, run, sample_targets)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

MAX_DEPTH = 4


@pytest.fixture(scope="module")
def provo_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("provo")
    _norms_rows().to_csv(d / "Provo_Corpus-Predictability_Norms.csv", index=False,
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
    """Provo plus the frozen candidate sets, exactly as the build writes them."""
    provo = load_provo(provo_dir, require_eye=True)
    cfg = BuildConfig(max_candidates=20, max_depth=MAX_DEPTH, top_k_expansions=0)
    words, keys = [], []
    build_corpus(provo, scorer, load_subtlex(None), cfg, keep_words=words, keep_keys=keys)
    return provo, keys, words


@pytest.fixture(scope="module")
def probe(built, scorer):
    provo, keys, words = built
    return prefix_restoration_probe(provo, keys, words, scorer, sample=len(keys),
                                    seed=DEFAULT_SEED, max_depth=MAX_DEPTH)


def _n_context(provo, tid, wn):
    return sum(1 for w in provo.passages[int(tid)][:int(wn)] if w)


def test_probe_covers_every_depth_of_every_sampled_target(built, probe):
    provo, keys, _ = built
    summary = probe["summary"]
    assert summary["n_targets_probed"] == len(keys) == summary["n_targets_available"]
    expected = sum(min(_n_context(provo, t, w), MAX_DEPTH) + 1 for t, w in keys)
    assert summary["n_pairs"] == expected
    assert len(probe["per_target"]) == len(keys)
    assert [d["depth"] for d in probe["by_depth"]] == list(range(MAX_DEPTH + 1))
    assert sum(d["n_targets"] for d in probe["by_depth"]) == expected
    assert probe["diagnostic"] == DIAGNOSTIC
    for d in probe["by_depth"]:
        assert 0.0 <= d["median_tv_gap"] <= d["max_tv_gap"] <= 1.0
    assert 0.0 <= summary["median_tv_gap"] <= summary["max_tv_gap"] <= 1.0
    assert summary["model"] == "test-tiny"


def test_a_prompt_equal_to_the_bare_span_scores_exactly_zero(built, probe):
    """Depth 0 and a target's own context length put the same string on both sides."""
    provo, keys, _ = built
    by_depth = {d["depth"]: d for d in probe["by_depth"]}
    assert by_depth[0]["n_identical_prompts"] == by_depth[0]["n_targets"]
    assert by_depth[0]["max_tv_gap"] == 0.0

    # Word 3 of a passage is the first target there and carries one word of
    # context, so its restored prefix is its bare span at both of its depths.
    first = [(t, w) for t, w in keys if w == FIRST_WORD_NUMBER + 1]
    assert first, "the fixture has no target at the start of a passage"
    for tid, wn in first:
        row = next(r for r in probe["per_target"]
                   if (r["text_id"], r["word_number"]) == (int(tid), int(wn)))
        assert row["K"] == 1 and row["median_tv_gap"] == 0.0


def test_the_truncated_depths_are_the_ones_that_move(probe):
    summary = probe["summary"]
    assert 0 < summary["n_truncated_pairs"] < summary["n_pairs"]
    assert summary["max_tv_gap"] > 0.0
    assert summary["median_tv_gap_truncated"] > 0.0


def test_gaps_are_zero_where_and_only_where_the_prompts_coincide(built, probe):
    """Which depths are exempt follows from the passage geometry, not from the model."""
    provo, keys, _ = built
    seen = 0
    for r in probe["pairs"]:
        n_ctx = _n_context(provo, r["text_id"], r["word_number"])
        # The bare span is the whole preceding context exactly when the depth
        # reaches it, and that context is the passage's own opening words.
        coincide = r["depth"] == 0 or r["depth"] == n_ctx
        assert r["identical_prompt"] is coincide
        assert (r["tv_gap"] == 0.0) is coincide
        seen += coincide
    assert seen > len(keys), "the depth-0 rows alone should not exhaust the exempt depths"


def test_the_same_seed_probes_the_same_targets_with_the_same_numbers(built, scorer):
    provo, keys, words = built
    a = prefix_restoration_probe(provo, keys, words, scorer, sample=4, seed=7,
                                 max_depth=MAX_DEPTH)
    b = prefix_restoration_probe(provo, keys, words, scorer, sample=4, seed=7,
                                 max_depth=MAX_DEPTH)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["summary"]["n_targets_probed"] == 4


def test_the_draw_is_seeded_sorted_and_a_subset():
    keys = [(1, i) for i in range(200)]
    first = sample_targets(keys, 12, seed=0)
    assert first == sorted(first) == sorted(set(first))
    assert len(first) == 12 and max(first) < 200
    assert first == sample_targets(keys, 12, seed=0)
    assert first != sample_targets(keys, 12, seed=1)
    # A sample larger than the corpus probes all of it rather than failing.
    assert sample_targets(keys[:5], 12, seed=0) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="at least one target"):
        sample_targets(keys, 0, seed=0)


def test_mismatched_targets_and_candidates_are_refused(built, scorer):
    provo, keys, words = built
    with pytest.raises(ValueError, match="candidate lists"):
        prefix_restoration_probe(provo, keys, words[:-1], scorer, sample=2)


def test_the_run_manifest_carries_the_median_gap(built, scorer, tmp_path):
    provo, keys, words = built
    rec = run(provo, keys, words, scorer, tmp_path, sample=3, seed=1, max_depth=MAX_DEPTH)
    assert (tmp_path / "prefix_probe.json").exists()
    assert (tmp_path / "prefix_probe.csv").exists()

    man = write_manifest(tmp_path)
    entry = man["diagnostics"][DIAGNOSTIC]
    assert entry["source"] == "prefix_probe.json"
    assert entry["median_tv_gap"] == pytest.approx(rec["summary"]["median_tv_gap"])
    assert entry["seed"] == 1 and entry["sample"] == 3
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["diagnostics"][DIAGNOSTIC] == entry
    # The diagnostics block is an addition; the file hashes are untouched.
    assert on_disk["n_files"] == 2


def test_manifest_ignores_json_that_declares_no_diagnostic(tmp_path):
    (tmp_path / "plain.json").write_text(json.dumps({"summary": {"x": 1}}))
    (tmp_path / "broken.json").write_text("{not json")
    assert write_manifest(tmp_path)["diagnostics"] == {}


def test_the_cli_exposes_the_probe_with_the_module_defaults():
    from lcsa.cli import build_parser, cmd_prefix_probe

    args = build_parser().parse_args(["prefix-probe", "--provo-dir", "somewhere"])
    assert args.func is cmd_prefix_probe
    assert args.sample == DEFAULT_SAMPLE == 50
    assert args.seed == DEFAULT_SEED == 0
    assert args.max_depth is None
    assert args.targets.endswith("targets.csv") and args.candidates.endswith("candidates.json")


def test_the_probe_matches_a_hand_written_total_variation(built, scorer):
    """One gap, recomputed from the two prompt strings without the probe's help."""
    provo, keys, words = built
    tid, wn = keys[-1]
    passage = provo.passages[int(tid)]
    head = [w for w in passage if w]
    depth = 2
    bare = " ".join([w for w in passage[:int(wn)] if w][-depth:])
    p = np.exp(scorer.candidate_logprobs(bare, words[-1]))
    q = np.exp(scorer.candidate_logprobs(" ".join(head[:depth]), words[-1]))
    want = 0.5 * float(np.abs(p - q).sum())

    rec = prefix_restoration_probe(provo, keys[-1:], words[-1:], scorer, sample=1,
                                   max_depth=MAX_DEPTH)
    got = next(d for d in rec["by_depth"] if d["depth"] == depth)
    assert got["n_targets"] == 1
    assert got["median_tv_gap"] == pytest.approx(want, abs=1e-12)


def test_the_probe_summary_reaches_the_e1_summary_the_scorecard_reads(built, scorer, tmp_path):
    """Prediction 12 is scored from ``e1_summary.json``, which the probe never writes."""
    from conftest import draw_true_delta, make_corpus

    from lcsa.cli import build_parser
    from lcsa.experiments.e1_exactness import run as run_e1
    from lcsa.likelihood import NAIVE
    from lcsa.registration import build_registration, score

    provo, keys, words = built
    rec = run(provo, keys, words, scorer, tmp_path, sample=3, seed=1, max_depth=MAX_DEPTH)
    corpus = draw_true_delta(make_corpus(n_targets=16, n_clusters=4, seed=17), 0.316, seed=19)
    res = run_e1(corpus, [NAIVE], tmp_path, prefix_probe=rec["summary"])
    assert res["prefix_probe"] == rec["summary"]
    on_disk = json.loads((tmp_path / "e1_summary.json").read_text())
    assert on_disk["prefix_probe"]["median_tv_gap_truncated"] == pytest.approx(
        rec["summary"]["median_tv_gap_truncated"])

    scored = {r["id"]: r for r in score(build_registration(legs=("e1",)), tmp_path)["predictions"]}
    assert scored[12]["status"] != "indeterminate"
    assert scored[12]["measured"]["median_tv_gap_truncated"] == pytest.approx(
        rec["summary"]["median_tv_gap_truncated"])
    # And the leg can be asked for it from the command line.
    assert build_parser().parse_args(["e1"]).prefix_probe is None
    assert build_parser().parse_args(["e1", "--prefix-probe", "p.json"]).prefix_probe == "p.json"
