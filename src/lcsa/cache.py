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
    """Context text before ``target_index`` retaining the last ``depth`` words.

    ``words`` is a passage indexed by Provo word_number, so ``target_index`` is
    the target's own word_number and the context is everything numbered below it.

    ``depth=None`` means the full preceding context.  ``depth=0`` gives the empty
    string, which is the row the truncation mixture weights most heavily as
    ``delta`` grows.
    """
    # Empty entries are the word numbers Provo does not carry, so they are
    # dropped before the depth slice and a depth of K retains K real words.
    prefix = [w for w in words[:target_index] if w]
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
        max_batch_tokens: int = 16384,
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
        # The reference path forwards every trie node as its own row, so a
        # 120-type candidate set is about sixty rows of up to 1,025 tokens.  The
        # model returns logits for every position of every row unless told
        # otherwise, and that tensor, not the weights, is what runs a 22 GB card
        # out of memory: 60 x 1025 x 152k x 2 bytes is 18.7 GB before the cast
        # to float32.  Rows are therefore forwarded in chunks under this token
        # budget with ``logits_to_keep=1`` where the model accepts it.
        self.max_batch_tokens = int(max_batch_tokens)
        self._logits_to_keep_ok = None
        self.path = path
        self._resolved = None
        # G1 is priced as 2N FLOP per parameter per token forwarded, so the
        # scorer counts tokens across every path and exposes N.
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        head = getattr(self.model, "get_output_embeddings", lambda: None)()
        if head is not None and hasattr(head, "weight"):
            self.vocab_size = int(head.weight.shape[0])
        else:
            self.vocab_size = int(getattr(getattr(self.model, "config", None), "vocab_size", 0)
                                  or getattr(self.tok, "vocab_size", 0) or 0)
        self.tokens_forwarded = 0

    # -- memory ---------------------------------------------------------

    def logits_bytes(self, tokens: int, kept: int | None = None) -> int:
        """Bytes the logits of one forward occupy: the half-precision tensor the
        model returns for ``tokens`` positions plus the float32 copy of the
        ``kept`` rows that the log-softmax runs on."""
        kept = tokens if kept is None else kept
        elt = self.torch.finfo(self.dtype).bits // 8
        return tokens * self.vocab_size * elt + kept * self.vocab_size * 4

    def memory_preflight(self, n_rows: int = 64) -> dict:
        """Estimate the peak logits allocation of each path against the device.

        The estimate is what the seed-noise sbatch comment ("any GPU type holds
        a 410M model") left out: the logits scale with tokens times vocabulary
        and not with the parameter count.  Returns the figures and raises when
        the packed path alone would not fit, so the job fails in its first
        second with the card named instead of hours in with an OOM trace.
        """
        torch = self.torch
        packed_tokens = self.max_prefix_tokens + 4 * n_rows
        packed = self.logits_bytes(packed_tokens, kept=4 * n_rows)
        chunk_rows = max(1, self.max_batch_tokens // (self.max_prefix_tokens + 8))
        simple = self.logits_bytes(chunk_rows * (self.max_prefix_tokens + 8), kept=chunk_rows)
        simple_all_positions = self.logits_bytes(chunk_rows * (self.max_prefix_tokens + 8))
        weights = self.n_params * (torch.finfo(self.dtype).bits // 8)
        rep = {"device": self.device, "vocab_size": self.vocab_size, "weights_bytes": weights,
               "packed_logits_bytes": packed, "simple_chunk_logits_bytes": simple,
               "simple_chunk_logits_bytes_without_logits_to_keep": simple_all_positions,
               "chunk_rows": chunk_rows, "total_bytes": None, "name": None}
        if self.device.startswith("cuda") and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.device(self.device))
            rep["total_bytes"] = int(props.total_memory)
            rep["name"] = props.name
            need = weights + max(packed, simple_all_positions)
            if need > 0.9 * props.total_memory:
                raise RuntimeError(
                    f"{props.name} has {props.total_memory / 2**30:.1f} GiB; weights "
                    f"({weights / 2**30:.1f} GiB) plus the logits of one forward "
                    f"({max(packed, simple_all_positions) / 2**30:.1f} GiB at vocabulary "
                    f"{self.vocab_size}) exceed 90 percent of it; request a 40 GB+ card "
                    "(--gres=gpu:l40s:1) or lower --max-batch-tokens")
        log.info("memory preflight: %s", rep)
        return rep

    # -- low level ------------------------------------------------------

    def passage_nll(self, words: Sequence[str]) -> tuple[float, int, np.ndarray]:
        """Summed token negative log-likelihood, token count, and per-token NLLs.

        The passage is scored as the model reads it, from the bos token if the
        tokeniser has one, in windows of ``max_prefix_tokens`` with the first
        token of each later window conditioned on the whole preceding window.
        Perplexity is ``exp(sum / count)`` over passages; it labels the
        competence confounds and is not a null of any kind.
        """
        torch = self.torch
        text = " ".join(str(w) for w in words)
        ids = self._prefix_ids("") + self.tok.encode(text, add_special_tokens=False)
        total, count, per_token = 0.0, 0, []
        step = self.max_prefix_tokens
        start = 0
        while start + 1 < len(ids):
            chunk = ids[start:start + step + 1]
            x = torch.tensor([chunk], dtype=torch.long, device=self.device)
            self.tokens_forwarded += int(x.numel())
            with torch.no_grad():
                logits = self.model(input_ids=x).logits[0, :-1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            tgt = x[0, 1:]
            nll = (-lp.gather(1, tgt[:, None])[:, 0]).double().cpu().numpy()
            per_token.append(nll)
            total += float(nll.sum())
            count += int(tgt.numel())
            start += step
        flat = np.concatenate(per_token) if per_token else np.zeros(0)
        return total, count, flat.astype(np.float64)

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

    # Transformers renamed the argument in 4.50; the older spelling is what pip
    # resolves under the compute nodes' Python 3.9, so both are tried once and
    # the working one is remembered for the rest of the build.
    _KEEP_KWARGS = ("logits_to_keep", "num_logits_to_keep")

    def _last_logits(self, ids, att, pos):
        """Logits at the final position only, asking the model to keep one row
        where it supports either spelling of the argument and slicing otherwise."""
        torch = self.torch
        with torch.no_grad():
            if self._logits_to_keep_ok is not False:
                names = ([self._logits_to_keep_ok] if isinstance(self._logits_to_keep_ok, str)
                         else self._KEEP_KWARGS)
                for name in names:
                    try:
                        out = self.model(input_ids=ids, attention_mask=att, position_ids=pos,
                                         **{name: 1}).logits
                    except TypeError:
                        continue
                    self._logits_to_keep_ok = name
                    return out[:, -1, :]
                self._logits_to_keep_ok = False
                log.warning("%s accepts neither %s; the reference path keeps every position's "
                            "logits and needs the memory for it",
                            self.model_name, " nor ".join(self._KEEP_KWARGS))
            return self.model(input_ids=ids, attention_mask=att, position_ids=pos).logits[:, -1, :]

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
        rows_per_chunk = max(1, self.max_batch_tokens // maxlen)
        for start in range(0, len(batch), rows_per_chunk):
            chunk = batch[start:start + rows_per_chunk]
            ids = torch.full((len(chunk), maxlen), pad, dtype=torch.long, device=self.device)
            att = torch.zeros((len(chunk), maxlen), dtype=torch.long, device=self.device)
            for r, s in enumerate(chunk):
                # Left padding keeps the final position at index maxlen-1 for all rows.
                ids[r, maxlen - len(s) :] = torch.tensor(s, dtype=torch.long, device=self.device)
                att[r, maxlen - len(s) :] = 1
            # Explicit position ids are mandatory under left padding: the default is
            # arange(maxlen), which would shift every short row's positions and make
            # this "reference" path quietly wrong.
            pos = (att.cumsum(dim=1) - 1).clamp(min=0)
            self.tokens_forwarded += int(ids.numel())
            lp = torch.log_softmax(self._last_logits(ids, att, pos).float(), dim=-1).cpu().numpy()
            for r, node in enumerate(keys[start:start + rows_per_chunk]):
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
        self.tokens_forwarded += int(ii.numel())
        with torch.no_grad():
            logits = self.model(input_ids=ii, attention_mask=add, position_ids=pp).logits
        # Only the evaluated nodes' rows are cast to float32 and normalised; the
        # prefix positions are never read, and casting the whole sequence would
        # double the largest allocation of the build for nothing.
        keep = list(trie.eval_nodes)
        idx = torch.tensor([rows[node] for node in keep], dtype=torch.long, device=self.device)
        lp = torch.log_softmax(logits[0].index_select(0, idx).float(), dim=-1).cpu().numpy()
        return {node: lp[k] for k, node in enumerate(keep)}

    # -- path selection -------------------------------------------------

    def resolve_path(self, probe_trie: CandidateTrie | None = None, tol: float = 5e-3) -> str:
        """Decide between ``packed`` and ``simple`` by comparing them on a probe."""
        if self._resolved is not None:
            return self._resolved
        if self.path == "simple":
            self._resolved = "simple"
            return self._resolved
        if probe_trie is None:
            # The packed forward's own risk is the position ids and attention
            # mask of the nodes below the root; a probe of single-token words
            # evaluates the root alone and would pass whatever they were.
            # The long words split under every tokenizer the build accepts,
            # and the space matches how the build encodes a candidate.
            words = ["the", "a", "an", "and", "unbelievably", "counterintuitively",
                     "photosynthesising", "antidisestablishmentarianism"]
            ids = [self.tok.encode(" " + w, add_special_tokens=False) for w in words]
            probe_trie = build_trie(ids, words)
        if probe_trie.max_depth() < 2:
            log.warning("the probe trie has no node below the root, so the packed path "
                        "check covers the first token position alone")
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
