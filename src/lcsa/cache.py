"""Building the nested ablation cache with a frozen reference language model.

For every cloze target we need ``K + 1`` distributions over the candidate set,
one per retained-context depth, where depth ``j`` conditions on the last ``j``
words.  Across Provo that is 61,435 distinct (target, depth) contexts, computed
exactly rather than sampled.

Two evaluation paths are implemented and the fast one is never trusted on faith.

``simple``
    One forward pass per (context, trie node).  Obviously correct, slow, and the
    reference the fast path is checked against.

``packed``
    One forward pass per context.  The prefix is laid down once, every trie node
    token is appended, and a 4-D additive attention mask lets each trie token
    see the prefix and its own ancestors only, with
    ``position_ids = len(prefix) + depth - 1``.  On a 120-type candidate set the
    trie has roughly 60 internal positions, so this replaces about 60 forwards
    with one.

``ReferenceScorer`` selects between them by *running both on a small sample and
comparing*, because a 4-D mask is accepted silently by some attention
implementations and ignored by others.  If the agreement check fails for any
reason the scorer falls back to ``simple`` and says so; nothing in the pipeline
depends on ``packed`` being available.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from lcsa.tokenization import CandidateTrie, build_trie, encode_candidates

log = logging.getLogger(__name__)

__all__ = ["ReferenceScorer", "context_string", "depth_contexts"]


def context_string(words: Sequence[str], target_index: int, depth: int | None) -> str:
    """Context text for a target at ``target_index`` retaining the last ``depth`` words.

    ``depth=None`` means the full preceding context.  ``depth=0`` gives the empty
    string, which is the row the truncation mixture weights most heavily as
    ``delta`` grows.
    """
    prefix = list(words[:target_index])
    if depth is not None:
        prefix = prefix[len(prefix) - depth :] if depth > 0 else []
    return " ".join(prefix)


def depth_contexts(words: Sequence[str], target_index: int, K: int) -> list[str]:
    """The ``K + 1`` context strings, index ``j`` retaining the last ``j`` words."""
    return [context_string(words, target_index, j) for j in range(K + 1)]


def _build_packed_inputs(prefix_ids, trie, pad_id, device, torch):
    """Return ``(input_ids, position_ids, mask4d, node_rows)`` for one context.

    ``node_rows`` maps a trie node index to the row of the output whose logits
    predict that node's next token: the last prefix position for the root, and
    the node's own position otherwise.
    """
    order = [i for i in range(1, trie.n_nodes)]  # every non-root node
    L = len(prefix_ids)
    n = len(order)
    ids = list(prefix_ids) + [trie.nodes[i].token for i in order]
    pos = list(range(L)) + [L + trie.nodes[i].depth - 1 for i in order]
    total = L + n

    allow = np.zeros((total, total), dtype=bool)
    # Causal block over the prefix.
    allow[:L, :L] = np.tril(np.ones((L, L), dtype=bool))
    slot = {node: L + k for k, node in enumerate(order)}
    for node in order:
        row = slot[node]
        allow[row, :L] = True
        for anc in trie.ancestors(node):
            allow[row, slot[anc]] = True
    node_rows = {0: L - 1}
    for node in order:
        node_rows[node] = slot[node]

    ii = torch.tensor([ids], dtype=torch.long, device=device)
    pp = torch.tensor([pos], dtype=torch.long, device=device)
    mm = torch.tensor(allow[None, None], dtype=torch.bool, device=device)
    return ii, pp, mm, node_rows


class ReferenceScorer:
    """Frozen reference model producing candidate log-probabilities per context."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        device: str | None = None,
        dtype: str = "float16",
        path: str = "auto",
        model=None,
        tokenizer=None,
        max_prefix_tokens: int = 1024,
    ) -> None:
        import torch  # imported lazily so the CPU-only estimation path needs no torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        want = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        if dtype not in want:
            raise ValueError(f"dtype must be one of {sorted(want)}, got {dtype!r}")
        # fp16 matmuls on CPU are slow and on some builds unimplemented.
        self.dtype = want[dtype] if device != "cpu" else torch.float32

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tok = tokenizer
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=self.dtype, attn_implementation="eager"
            )
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.max_prefix_tokens = int(max_prefix_tokens)
        self.path = path
        self._resolved = None

    # -- low level ------------------------------------------------------

    def _prefix_ids(self, context: str) -> list[int]:
        if not context:
            bos = getattr(self.tok, "bos_token_id", None)
            if bos is None:
                bos = getattr(self.tok, "eos_token_id", None)
            if bos is None:
                raise ValueError(
                    f"{self.model_name} has neither a bos nor an eos token, so the "
                    "empty context (depth 0) cannot be scored"
                )
            return [int(bos)]
        ids = self.tok.encode(context, add_special_tokens=False)
        if not ids:
            raise ValueError(f"context {context[:40]!r} encoded to nothing")
        return ids[-self.max_prefix_tokens :]

    def _simple_nodes(self, prefix_ids: list[int], trie: CandidateTrie) -> dict[int, np.ndarray]:
        torch = self.torch
        out: dict[int, np.ndarray] = {}
        batch, keys = [], []
        for node in trie.eval_nodes:
            seq = prefix_ids + [trie.nodes[i].token for i in trie.ancestors(node)]
            batch.append(seq)
            keys.append(node)
        maxlen = max(len(s) for s in batch)
        pad = int(self.tok.pad_token_id)
        ids = torch.full((len(batch), maxlen), pad, dtype=torch.long, device=self.device)
        att = torch.zeros((len(batch), maxlen), dtype=torch.long, device=self.device)
        for r, s in enumerate(batch):
            # Left padding keeps the final position at index maxlen-1 for all rows.
            ids[r, maxlen - len(s) :] = torch.tensor(s, dtype=torch.long, device=self.device)
            att[r, maxlen - len(s) :] = 1
        # Explicit position ids are mandatory under left padding: the default is
        # arange(maxlen), which would shift every short row's positions and make
        # this "reference" path quietly wrong.
        pos = (att.cumsum(dim=1) - 1).clamp(min=0)
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=att, position_ids=pos).logits
        lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1).cpu().numpy()
        for r, node in enumerate(keys):
            out[node] = lp[r]
        return out

    def _packed_nodes(self, prefix_ids: list[int], trie: CandidateTrie) -> dict[int, np.ndarray]:
        torch = self.torch
        ii, pp, mm, rows = _build_packed_inputs(
            prefix_ids, trie, int(self.tok.pad_token_id), self.device, torch
        )
        neg = torch.finfo(self.dtype).min
        add = torch.where(mm, torch.zeros((), dtype=self.dtype, device=self.device),
                          torch.full((), neg, dtype=self.dtype, device=self.device))
        with torch.no_grad():
            logits = self.model(input_ids=ii, attention_mask=add, position_ids=pp).logits
        lp = torch.log_softmax(logits[0].float(), dim=-1).cpu().numpy()
        return {node: lp[rows[node]] for node in trie.eval_nodes}

    # -- path selection -------------------------------------------------

    def resolve_path(self, probe_trie: CandidateTrie | None = None, tol: float = 5e-3) -> str:
        """Decide between ``packed`` and ``simple`` by comparing them on a probe."""
        if self._resolved is not None:
            return self._resolved
        if self.path == "simple":
            self._resolved = "simple"
            return self._resolved
        if probe_trie is None:
            ids = [self.tok.encode(w, add_special_tokens=False) for w in ["the", "a", "an", "and"]]
            probe_trie = build_trie(ids, ["the", "a", "an", "and"])
        prefix = self._prefix_ids("the quick brown fox jumps over the lazy")
        try:
            a = self._simple_nodes(prefix, probe_trie)
            b = self._packed_nodes(prefix, probe_trie)
            err = max(
                float(np.abs(a[k] - b[k]).max()) for k in probe_trie.eval_nodes
            )
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            log.warning("packed path unavailable (%s: %s); using simple", type(exc).__name__, exc)
            self._resolved = "simple"
            return self._resolved
        if not math.isfinite(err) or err > tol:
            log.warning(
                "packed path disagrees with simple by %.3g > %.3g; using simple", err, tol
            )
            self._resolved = "simple"
        else:
            log.info("packed path verified against simple, max log-prob error %.3g", err)
            self._resolved = "packed"
        if self.path == "packed" and self._resolved == "simple":
            raise RuntimeError(
                "path='packed' was requested but the agreement check failed; "
                "rerun with path='auto' to fall back or path='simple' to force the "
                "reference implementation"
            )
        return self._resolved

    # -- public API -----------------------------------------------------

    def node_logprobs(self, context: str, trie: CandidateTrie) -> dict[int, np.ndarray]:
        prefix = self._prefix_ids(context)
        path = self.resolve_path()
        if path == "packed":
            return self._packed_nodes(prefix, trie)
        return self._simple_nodes(prefix, trie)

    def candidate_logprobs(
        self, context: str, words: Sequence[str], oov_index: int | None = None
    ) -> np.ndarray:
        """``log p(word | context)`` for each candidate, trie-marginalised."""
        from lcsa.tokenization import OOV, word_log_probs

        seqs = encode_candidates(self.tok, words, context)
        trie = build_trie(seqs, list(words))
        if oov_index is None and OOV in words:
            oov_index = list(words).index(OOV)
        nl = self.node_logprobs(context, trie)
        return word_log_probs(trie, nl, oov_index=oov_index)

    def target_cache(
        self,
        words: Sequence[str],
        target_index: int,
        candidates: Sequence[str],
        K: int,
        oov_index: int | None = None,
    ) -> np.ndarray:
        """The ``(K + 1, V)`` nested ablation block for one target, as probabilities."""
        rows = []
        for ctx in depth_contexts(words, target_index, K):
            lp = self.candidate_logprobs(ctx, candidates, oov_index)
            rows.append(np.exp(lp - lp.max()) * math.exp(float(lp.max())))
        P = np.asarray(rows, dtype=np.float64)
        if not np.all(np.isfinite(P)):
            raise FloatingPointError(
                f"non-finite cache row for target {target_index} of a "
                f"{len(words)}-word passage"
            )
        return P
