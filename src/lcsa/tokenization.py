"""Word probabilities from a subword language model, by prefix-trie marginalisation.

A cloze response is a *word*, and a language model scores *tokens*, so every
number in this paper depends on getting the conversion right.  Two errors are
routine and both are fatal here, because a mistake propagates into every cached
distribution and no downstream statistic can detect it (Remark 5 of the paper).

1.  Leading whitespace.  In byte-level BPE, ``"dog"`` and ``" dog"`` are
    different token sequences with very different probabilities, and mid-sentence
    continuations take the space-prefixed form.  We decide the form from the
    context string, following Oh and Schuler (2024) and Pimentel and Meister
    (2024), and we assert the decoded round trip.
2.  Shared prefixes.  Two candidates that begin with the same token must not be
    scored with two independent forward passes at different numerical values of
    the same conditional.  Building a trie makes the shared conditionals shared
    by construction and cuts the number of forward positions by roughly half on
    a 120-type candidate set.

The trie built here is consumed by :mod:`lcsa.cache`, which evaluates all its
internal nodes in one packed forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "TrieNode",
    "CandidateTrie",
    "build_trie",
    "encode_candidates",
    "needs_leading_space",
    "word_log_probs",
    "OOV",
]

#: Sentinel candidate absorbing the mass outside the observed candidate set.
OOV = "<oov>"


def needs_leading_space(context: str) -> bool:
    """True when the next word should be encoded with a leading space.

    False at the start of a passage and after an existing trailing space, which
    is the only case where the space-free form is correct.
    """
    return bool(context) and not context[-1].isspace()


def encode_candidates(
    tokenizer,
    words: Sequence[str],
    context: str,
    verify: bool = True,
) -> list[list[int]]:
    """Token id sequences for each candidate word under the context's spacing rule.

    Raises on an empty encoding or a failed decode round trip rather than
    returning something plausible, because a silent tokenisation error is the
    one failure mode this pipeline cannot detect downstream.
    """
    space = needs_leading_space(context)
    out: list[list[int]] = []
    for w in words:
        if w == OOV:
            out.append([])
            continue
        surface = (" " + w) if space else w
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if not ids:
            raise ValueError(f"candidate {w!r} encoded to an empty token sequence")
        if verify:
            back = tokenizer.decode(ids)
            if back.strip() != surface.strip():
                raise ValueError(
                    f"tokenisation round trip failed for {w!r}: "
                    f"encoded {surface!r} -> {ids} -> decoded {back!r}"
                )
        out.append(list(ids))
    return out


@dataclass
class TrieNode:
    token: int  # token id on the edge into this node (-1 at the root)
    depth: int  # 0 at the root
    parent: int  # index of the parent node (-1 at the root)
    children: dict[int, int] = field(default_factory=dict)
    terminal_for: list[int] = field(default_factory=list)


@dataclass
class CandidateTrie:
    """Prefix trie over candidate token sequences.

    Attributes
    ----------
    nodes
        Node 0 is the root.  A node's ``depth`` gives its position offset in the
        packed forward, ``position_ids = len(prefix) + depth - 1``.
    eval_nodes
        Nodes whose next-token distribution is actually needed: the root plus
        every node with children.  Leaves need no forward pass.
    """

    nodes: list[TrieNode]
    eval_nodes: list[int]
    sequences: list[list[int]]
    words: list[str]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_eval(self) -> int:
        return len(self.eval_nodes)

    def ancestors(self, i: int) -> list[int]:
        """Node indices from the root's child down to ``i`` inclusive."""
        chain: list[int] = []
        while i > 0:
            chain.append(i)
            i = self.nodes[i].parent
        chain.reverse()
        return chain

    def max_depth(self) -> int:
        return max((n.depth for n in self.nodes), default=0)


def build_trie(sequences: Sequence[Sequence[int]], words: Sequence[str]) -> CandidateTrie:
    """Build the trie and mark which nodes need a forward pass."""
    if len(sequences) != len(words):
        raise ValueError(
            f"{len(sequences)} sequences for {len(words)} words"
        )
    nodes = [TrieNode(token=-1, depth=0, parent=-1)]
    for ci, seq in enumerate(sequences):
        cur = 0
        for tok in seq:
            nxt = nodes[cur].children.get(int(tok))
            if nxt is None:
                nxt = len(nodes)
                nodes.append(
                    TrieNode(token=int(tok), depth=nodes[cur].depth + 1, parent=cur)
                )
                nodes[cur].children[int(tok)] = nxt
            cur = nxt
        nodes[cur].terminal_for.append(ci)
    eval_nodes = [i for i, n in enumerate(nodes) if n.children]
    if 0 not in eval_nodes:
        eval_nodes.insert(0, 0)
    eval_nodes.sort()
    return CandidateTrie(
        nodes=nodes,
        eval_nodes=eval_nodes,
        sequences=[list(s) for s in sequences],
        words=list(words),
    )


def word_log_probs(
    trie: CandidateTrie,
    node_logprobs: dict[int, np.ndarray],
    oov_index: int | None = None,
) -> np.ndarray:
    """Assemble ``log p(word | context)`` for every candidate.

    Parameters
    ----------
    node_logprobs
        Maps an evaluated node index to its full next-token log-probability
        vector, length ``|vocab|``.  Only ``trie.eval_nodes`` need be present.
    oov_index
        Index of the out-of-vocabulary bucket in ``trie.words``, which receives
        the leftover mass ``log(max(1 - sum exp(lp), floor))``.  The bucket keeps
        the candidate set a genuine probability distribution instead of a
        renormalised one, so that a target whose responses are mostly outside
        the set does not silently inflate every candidate.
    """
    n = len(trie.words)
    lp = np.full(n, -np.inf, dtype=np.float64)
    for ci, seq in enumerate(trie.sequences):
        if oov_index is not None and ci == oov_index:
            continue
        total = 0.0
        cur = 0
        ok = True
        for tok in seq:
            vec = node_logprobs.get(cur)
            if vec is None:
                raise KeyError(
                    f"node {cur} (depth {trie.nodes[cur].depth}) was never evaluated, "
                    f"but candidate {trie.words[ci]!r} needs it"
                )
            total += float(vec[int(tok)])
            cur = trie.nodes[cur].children[int(tok)]
            if not np.isfinite(total):
                ok = False
                break
        lp[ci] = total if ok else -np.inf
    if oov_index is not None:
        mass = float(np.exp(lp[np.isfinite(lp)]).sum())
        lp[oov_index] = float(np.log(max(1.0 - mass, 1e-12)))
    return lp


def check_no_duplicates(words: Sequence[str]) -> None:
    """Fail loudly on a duplicated candidate; two identical rows break the counts."""
    seen: set[str] = set()
    dup = [w for w in words if (w in seen) or seen.add(w)]  # type: ignore[func-returns-value]
    if dup:
        raise ValueError(f"duplicate candidates in the set: {sorted(set(dup))}")
