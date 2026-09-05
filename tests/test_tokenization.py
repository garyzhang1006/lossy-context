"""Candidate encoding, the prefix trie, and word-level marginalisation.

A silent tokenisation error is the one failure this pipeline cannot detect
downstream (Remark 5 of the paper), so these tests are about loud failure as
much as about correct arithmetic.  The tokenizer here is a byte-level stand-in,
which keeps the suite free of any model download.
"""

from __future__ import annotations

import numpy as np
import pytest

from lcsa.tokenization import (OOV, build_trie, check_no_duplicates,
                               encode_candidates, needs_leading_space,
                               word_log_probs)


class ByteTokenizer:
    """Byte-level tokenizer: one token per byte, so the trie shape is predictable."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 256

    def encode(self, text, add_special_tokens=False):
        return [int(b) for b in text.encode("utf-8")]

    def decode(self, ids):
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


class LossyTokenizer(ByteTokenizer):
    """Drops the last byte on decode, which is exactly the corruption G0 hunts for."""

    def decode(self, ids):
        return super().decode(list(ids)[:-1])


def test_leading_space_rule():
    assert needs_leading_space("the cat sat on")
    assert not needs_leading_space("")
    assert not needs_leading_space("the cat ")


def test_candidates_get_a_leading_space_only_mid_passage():
    tok = ByteTokenizer()
    mid = encode_candidates(tok, ["dog"], "the cat chased the")
    start = encode_candidates(tok, ["dog"], "")
    assert tok.decode(mid[0]) == " dog"
    assert tok.decode(start[0]) == "dog"
    assert mid[0] != start[0]


def test_oov_bucket_encodes_to_nothing():
    """The bucket carries leftover mass and is never scored as a token sequence."""
    assert encode_candidates(ByteTokenizer(), [OOV], "context here") == [[]]


def test_empty_encoding_is_refused():
    with pytest.raises(ValueError, match="empty token sequence"):
        encode_candidates(ByteTokenizer(), [""], "")


def test_failed_round_trip_is_refused():
    with pytest.raises(ValueError, match="round trip failed"):
        encode_candidates(LossyTokenizer(), ["dog"], "the cat chased the")


def test_verification_can_be_switched_off_but_is_on_by_default():
    assert encode_candidates(LossyTokenizer(), ["dog"], "x y", verify=False)


def test_trie_merges_shared_prefixes():
    seqs = [[1, 2, 3], [1, 2, 4], [1, 5], [9]]
    trie = build_trie(seqs, ["a", "b", "c", "d"])
    # root, 1, 12, 123, 124, 15, 9
    assert trie.n_nodes == 7
    assert trie.max_depth() == 3
    # Only the root and nodes with children need a forward pass.
    assert set(trie.eval_nodes) == {0, 1, 2}
    assert trie.n_eval == 3


def test_trie_ancestor_chain_is_root_to_node():
    trie = build_trie([[7, 8, 9]], ["w"])
    leaf = 3
    assert [trie.nodes[i].token for i in trie.ancestors(leaf)] == [7, 8, 9]
    assert trie.ancestors(0) == []


def test_trie_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="sequences for"):
        build_trie([[1], [2]], ["only-one"])


def test_word_log_probs_applies_the_chain_rule():
    seqs = [[1, 2], [1, 3], [4]]
    trie = build_trie(seqs, ["ab", "ac", "d"])
    V = 8
    root = np.log(np.array([0.1, 0.5, 0.1, 0.1, 0.2, 0.0, 0.0, 0.0]) + 1e-300)
    after1 = np.log(np.array([0.1, 0.1, 0.3, 0.4, 0.1, 0.0, 0.0, 0.0]) + 1e-300)
    node_lp = {0: root, 1: after1}
    lp = word_log_probs(trie, node_lp)
    assert lp[0] == pytest.approx(np.log(0.5) + np.log(0.3), rel=1e-9)
    assert lp[1] == pytest.approx(np.log(0.5) + np.log(0.4), rel=1e-9)
    assert lp[2] == pytest.approx(np.log(0.2), rel=1e-9)


def test_oov_bucket_takes_the_leftover_mass():
    trie = build_trie([[1], [2], []], ["a", "b", OOV])
    root = np.log(np.array([0.05, 0.3, 0.25, 0.4]))
    lp = word_log_probs(trie, {0: root}, oov_index=2)
    assert np.exp(lp).sum() == pytest.approx(1.0, abs=1e-12)
    assert lp[2] == pytest.approx(np.log(1.0 - 0.3 - 0.25), rel=1e-9)


def test_oov_mass_never_goes_negative():
    """Numerical slop must not produce a log of a negative number."""
    trie = build_trie([[1], [2], []], ["a", "b", OOV])
    root = np.log(np.array([1e-9, 0.6, 0.4, 1e-9]))
    lp = word_log_probs(trie, {0: root}, oov_index=2)
    assert np.isfinite(lp[2])
    assert lp[2] < np.log(1e-6)


def test_missing_node_is_reported_not_guessed():
    trie = build_trie([[1, 2]], ["ab"])
    with pytest.raises(KeyError, match="never evaluated"):
        word_log_probs(trie, {0: np.zeros(4)})


def test_impossible_candidate_gets_minus_infinity():
    trie = build_trie([[1], [2]], ["a", "b"])
    root = np.array([-np.inf, -np.inf, np.log(1.0), -np.inf])
    lp = word_log_probs(trie, {0: root})
    assert lp[0] == -np.inf
    assert lp[1] == pytest.approx(0.0)


def test_duplicate_candidates_are_refused():
    check_no_duplicates(["a", "b", "c"])
    with pytest.raises(ValueError, match="duplicate candidates"):
        check_no_duplicates(["a", "b", "a"])


def test_word_probabilities_sum_to_at_most_one_on_a_real_shaped_set():
    """Marginalising a trie can never manufacture probability mass."""
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(30)]
    tok = ByteTokenizer()
    seqs = encode_candidates(tok, words, "some context")
    trie = build_trie(seqs, words)
    node_lp = {}
    for node in trie.eval_nodes:
        v = rng.dirichlet(np.ones(256))
        node_lp[node] = np.log(v)
    lp = word_log_probs(trie, node_lp)
    assert np.exp(lp).sum() <= 1.0 + 1e-9
