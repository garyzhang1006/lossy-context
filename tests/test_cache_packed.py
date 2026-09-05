"""The packed forward against the reference forward, on a real transformer.

The packed path replaces roughly sixty forwards per context with one, using a
4-D attention mask that some attention implementations accept and ignore.  These
tests run a genuine two-layer transformer with random weights, so the comparison
is against real attention rather than against a mock.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_tokenization import ByteTokenizer

from lcsa.cache import context_string, depth_contexts
from lcsa.tokenization import OOV, build_trie, encode_candidates

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

WORDS = ["the", "cat", "car", "cart", "dog", "door", "a", OOV]
CONTEXT = "the quick brown fox jumps over the lazy"


@pytest.fixture(scope="module")
def scorer():
    from transformers import GPT2Config, GPT2LMHeadModel

    from lcsa.cache import ReferenceScorer

    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=256, n_positions=256, n_embd=32, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(cfg)
    model.config._attn_implementation = "eager"
    return ReferenceScorer("test-tiny", device="cpu", dtype="float32",
                           model=model, tokenizer=ByteTokenizer())


@pytest.fixture(scope="module")
def trie():
    return build_trie(encode_candidates(ByteTokenizer(), WORDS, CONTEXT), WORDS)


def _brute_force(scorer, prefix_ids, trie):
    """One unpadded forward per node, which is as simple as this can be written."""
    out = {}
    for node in trie.eval_nodes:
        seq = list(prefix_ids) + [trie.nodes[i].token for i in trie.ancestors(node)]
        ids = torch.tensor([seq], dtype=torch.long)
        with torch.no_grad():
            logits = scorer.model(input_ids=ids).logits
        out[node] = torch.log_softmax(logits[0, -1].float(), dim=-1).numpy()
    return out


def test_reference_path_matches_an_unpadded_forward(scorer, trie):
    """Left padding plus explicit position ids must change nothing at all."""
    prefix = scorer._prefix_ids(CONTEXT)
    padded = scorer._simple_nodes(prefix, trie)
    brute = _brute_force(scorer, prefix, trie)
    err = max(float(np.abs(padded[k] - brute[k]).max()) for k in trie.eval_nodes)
    assert err < 1e-4


def test_packed_path_matches_the_reference_path(scorer, trie):
    """The 4-D mask must isolate each trie branch, or the whole cache is wrong."""
    prefix = scorer._prefix_ids(CONTEXT)
    a = scorer._simple_nodes(prefix, trie)
    b = scorer._packed_nodes(prefix, trie)
    err = max(float(np.abs(a[k] - b[k]).max()) for k in trie.eval_nodes)
    assert err < 5e-3, f"packed and simple disagree by {err:.3g} log units"


def test_path_resolution_picks_packed_when_it_agrees(scorer):
    assert scorer.resolve_path() == "packed"


def test_forcing_packed_when_it_disagrees_raises():
    """A silently ignored mask must stop the run rather than produce a wrong cache."""
    from transformers import GPT2Config, GPT2LMHeadModel

    from lcsa.cache import ReferenceScorer

    torch.manual_seed(1)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=256, n_positions=256, n_embd=16,
                                       n_layer=1, n_head=2))

    class IgnoresTheMask(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids=None, attention_mask=None, position_ids=None):
            return self.inner(input_ids=input_ids)

        def to(self, *a, **k):
            self.inner.to(*a, **k)
            return self

        def eval(self):
            self.inner.eval()
            return self

    s = ReferenceScorer("test-tiny", device="cpu", dtype="float32",
                        model=IgnoresTheMask(model), tokenizer=ByteTokenizer(),
                        path="packed")
    with pytest.raises(RuntimeError, match="agreement check failed"):
        s.resolve_path()


def test_candidate_log_probs_are_a_sub_distribution(scorer):
    lp = scorer.candidate_logprobs(CONTEXT, WORDS)
    assert lp.shape == (len(WORDS),)
    assert np.exp(lp).sum() == pytest.approx(1.0, abs=1e-6)


def test_empty_context_is_scored_through_the_bos_token(scorer):
    lp = scorer.candidate_logprobs("", WORDS)
    assert np.all(np.isfinite(lp))


def test_target_cache_has_one_row_per_depth(scorer):
    words = CONTEXT.split() + ["ends"]
    P = scorer.target_cache(words, len(words) - 1, WORDS, K=4)
    assert P.shape == (5, len(WORDS))
    assert np.all(P > 0)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)


def test_deepest_cache_row_is_the_full_context(scorer):
    """Row K must equal scoring the full retained context directly."""
    words = CONTEXT.split() + ["ends"]
    K = 4
    P = scorer.target_cache(words, len(words) - 1, WORDS, K=K)
    direct = np.exp(scorer.candidate_logprobs(context_string(words, len(words) - 1, K), WORDS))
    assert np.max(np.abs(P[K] - direct)) < 1e-9


def test_context_strings_retain_the_last_j_words():
    words = "a b c d e".split()
    assert context_string(words, 4, None) == "a b c d"
    assert context_string(words, 4, 2) == "c d"
    assert context_string(words, 4, 0) == ""
    ctx = depth_contexts(words, 4, 4)
    assert ctx == ["", "d", "c d", "b c d", "a b c d"]


def test_depth_beyond_the_passage_start_is_the_whole_prefix():
    words = "a b c".split()
    assert context_string(words, 2, 9) == "a b"
