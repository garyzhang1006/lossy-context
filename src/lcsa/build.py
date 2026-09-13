"""Turning Provo plus a reference model into the :class:`~lcsa.corpusdata.Corpus`.

This is the step Remark 5 of the paper calls the failure mode that could not be
designed away: the candidate set is frozen from the observed response types
*before* the nested cache is built, so a decode error or a misaligned word index
would propagate into every cached distribution with nothing downstream to catch
it.  Gate G0 lives here, and it is checked, not assumed.

Candidate set for a target, capped at 120 types:

    observed response types
  + the corpus target word
  + top-50 first-token expansions under the full context
  + an out-of-vocabulary bucket

Production features ``f(w)``: log unigram frequency, character length (scaled),
a content-word indicator, and log document frequency measured as the number of
Provo passages containing the word.  ``g(w, c)`` indicates that ``w`` already
occurred earlier in the passage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from lcsa.cache import context_string
from lcsa.corpusdata import Corpus
from lcsa.data.provo import ProvoData, canonical_word
from lcsa.data.subtlex import Unigrams
from lcsa.tokenization import OOV, check_no_duplicates

log = logging.getLogger(__name__)

__all__ = ["BuildConfig", "candidate_set", "feature_matrix", "buildable_targets",
           "build_corpus", "FEATURE_NAMES"]

FEATURE_NAMES = ["log_unigram", "length", "is_content", "log_docfreq"]


@dataclass
class BuildConfig:
    max_candidates: int = 120
    max_depth: int = 32
    top_k_expansions: int = 50
    min_responses: int = 1
    include_oov: bool = True


def _expansion_words(scorer, context: str, k: int) -> list[str]:
    """Words whose first token is among the top ``k`` next tokens under ``context``.

    Tokens that do not decode to a bare alphabetic word are dropped, since a
    punctuation or subword fragment is not a cloze response and would only
    consume room in a capped candidate set.
    """
    if scorer is None or k <= 0:
        return []
    import torch

    ids = scorer._prefix_ids(context)
    with torch.no_grad():
        logits = scorer.model(
            input_ids=torch.tensor([ids], device=scorer.device),
            position_ids=torch.arange(len(ids), device=scorer.device)[None],
        ).logits[0, -1]
    top = torch.topk(logits.float(), k=min(k, logits.shape[-1])).indices.tolist()
    out = []
    for t in top:
        s = scorer.tok.decode([int(t)]).strip()
        if s and s.isalpha():
            out.append(s.lower())
    return out


def candidate_set(
    responses: list[str],
    target_word: str,
    expansions: list[str],
    cfg: BuildConfig,
) -> list[str]:
    """Frozen candidate list for one target, response types first.

    Response types come first so that the cap never removes a word somebody
    actually produced; expansions fill whatever room is left.
    """
    seen: dict[str, None] = {}
    for w in responses:
        c = canonical_word(w)
        if c:
            seen.setdefault(c, None)
    t = canonical_word(target_word)
    if t:
        seen.setdefault(t, None)
    room = cfg.max_candidates - (1 if cfg.include_oov else 0)
    for w in expansions:
        if len(seen) >= room:
            break
        c = canonical_word(w)
        if c:
            seen.setdefault(c, None)
    words = list(seen)[:room]
    if cfg.include_oov:
        words.append(OOV)
    check_no_duplicates(words)
    return words


def feature_matrix(
    words: list[str],
    unigrams: Unigrams,
    content_words: set[str],
    doc_freq: dict[str, int],
    n_docs: int,
) -> np.ndarray:
    """``(V, 4)`` production features, standardised within target.

    Standardising within target keeps the ``kappa`` coordinates comparable
    across targets with very different candidate sets, and leaves the span of
    the nuisance directions unchanged, which is all the absorption criterion
    depends on.
    """
    n = len(words)
    F = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    oov_rows = []
    for i, w in enumerate(words):
        if w == OOV:
            oov_rows.append(i)
            continue
        F[i, 0] = unigrams(w)
        F[i, 1] = len(w)
        F[i, 2] = 1.0 if w in content_words else 0.0
        F[i, 3] = np.log(max(doc_freq.get(w, 0), 1) / max(n_docs, 1))
    if oov_rows:
        # The bucket is the least frequent, shortest, least widely attested
        # thing on the list; giving it zeros would instead place it at the mean
        # of the frequency features and let kappa borrow its mass.
        real = [i for i in range(n) if i not in oov_rows]
        floor_row = F[real].min(axis=0) if real else np.zeros(len(FEATURE_NAMES))
        for i in oov_rows:
            F[i] = floor_row
    sd = F.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (F - F.mean(axis=0)) / sd


def buildable_targets(provo: ProvoData, cfg: BuildConfig | None = None) -> list:
    """Every target the build admits, decided before any model is loaded.

    Returns ``(row, responses, word_number, n_context, K)`` per target, in the
    order :func:`build_corpus` builds them.  It is the same filter, called from
    the same place, so a login-node check can count the targets a submission
    will produce and compare that count against the registered design without
    holding a GPU for the answer.
    """
    cfg = cfg or BuildConfig()
    grouped = {k: g for k, g in provo.responses.groupby(["text_id", "word_number"])}
    out = []
    for r in provo.words.itertuples():
        grp = grouped.get((int(r.text_id), int(r.word_number)))
        if grp is None or float(grp["count"].sum()) < cfg.min_responses:
            continue
        passage = provo.passages[int(r.text_id)]
        # Passages are indexed by word_number, whose empty slots are the numbers
        # Provo does not carry, so the depth counts the real words below the
        # target rather than the index itself: three passages are missing a word
        # in the middle, and every passage is missing its first.
        ti = int(r.word_number)
        if not (0 < ti < len(passage)) or not passage[ti]:
            continue
        n_context = sum(1 for x in passage[:ti] if x)
        if n_context == 0:
            continue
        out.append((r, grp, ti, n_context, min(n_context, cfg.max_depth)))
    return out


def build_corpus(
    provo: ProvoData,
    scorer,
    unigrams: Unigrams,
    cfg: BuildConfig | None = None,
    limit: int | None = None,
    progress=None,
    select=None,
    keep_words: list | None = None,
    keep_keys: list | None = None,
    candidates: dict | None = None,
) -> Corpus:
    """Build the full nested cache.  This is the GPU-heavy step of the pipeline.

    ``select`` is an optional predicate on ``(text_id, word_number, K)`` used to
    restrict the build, which is how the bridging experiment gets the short
    contexts without paying for the other 2,248 targets.  ``keep_words``, if
    given, is appended with the candidate list of every target that was built,
    in the same order as the corpus, so that a caller needing the strings does
    not have to reconstruct them and risk a different candidate set.

    ``candidates`` freezes the candidate list per ``(text_id, word_number)``:
    targets absent from it are skipped and the expansion step is not run, so a
    cache built under another checkpoint has row for row the same candidate
    sets as the primary build and can stand in for it as a reference.
    """
    cfg = cfg or BuildConfig()

    doc_freq: dict[str, int] = {}
    for tid, ws in provo.passages.items():
        for w in {canonical_word(x) for x in ws}:
            if w:
                doc_freq[w] = doc_freq.get(w, 0) + 1
    n_docs = len(provo.passages)
    content_words = {
        canonical_word(r.word)
        for r in provo.words.itertuples()
        if getattr(r, "is_content", 0.0) == 1.0
    }
    # An is_content the loader could recover from neither arm leaves this set
    # empty, which makes the third lexical feature identically zero and drops
    # the repaired nuisance dimension without anything downstream noticing.
    if not content_words:
        log.warning("no target is marked as a content word, so the is_content "
                    "feature is constant and the repaired fit loses a dimension")

    P_list, n_list, u_list, f_list, g_list, clusters, wid, slots, nctx, tkeys = (
        [], [], [], [], [], [], [], [], [], []
    )
    n_done = 0
    for r, grp, ti, n_context, K in buildable_targets(provo, cfg):
        key = (int(r.text_id), int(r.word_number))
        passage = provo.passages[int(r.text_id)]
        if select is not None and not select(int(r.text_id), int(r.word_number), K):
            continue

        if candidates is not None:
            frozen = candidates.get(key)
            if frozen is None:
                continue
            words = [str(w) for w in frozen]
        else:
            full_ctx = context_string(passage, ti, None)
            expansions = _expansion_words(scorer, full_ctx, cfg.top_k_expansions)
            words = candidate_set(
                [str(x) for x in grp["response"].tolist()], str(r.word), expansions, cfg
            )
        index = {w: i for i, w in enumerate(words)}

        counts = np.zeros(len(words), dtype=np.float64)
        oov_i = index.get(OOV)
        for resp_word, c in zip(grp["response"].tolist(), grp["count"].tolist()):
            j = index.get(canonical_word(resp_word))
            if j is None:
                if oov_i is not None:
                    counts[oov_i] += float(c)
            else:
                counts[j] += float(c)

        P = scorer.target_cache(passage, ti, words, K, oov_index=oov_i)
        if P.shape != (K + 1, len(words)):
            raise ValueError(
                f"target {key}: cache has shape {P.shape}, expected {(K + 1, len(words))}"
            )

        prior = {canonical_word(x) for x in passage[:ti] if x}
        P_list.append(P)
        n_list.append(counts)
        u_list.append(unigrams.vector(words))
        f_list.append(feature_matrix(words, unigrams, content_words, doc_freq, n_docs))
        g_list.append(np.array([1.0 if w in prior else 0.0 for w in words]))
        clusters.append(int(r.text_id))
        wid.append(np.arange(len(words)))
        slots.append(index.get(canonical_word(str(r.word)), -1))
        # Kept beside K so a later reader can see which targets the cap actually
        # cut: prediction 13 needs an arm where it cut none of them.
        nctx.append(int(n_context))
        # The same pair `keep_keys` hands back and `targets.csv` lists, carried
        # on the corpus as well so an artifact built beside the cache can be
        # resolved against it without the csv being at hand.
        tkeys.append([int(r.text_id), int(r.word_number)])
        if keep_words is not None:
            keep_words.append(list(words))
        if keep_keys is not None:
            keep_keys.append((int(r.text_id), int(r.word_number)))

        n_done += 1
        if progress is not None:
            progress(n_done)
        if limit is not None and n_done >= limit:
            break

    if not P_list:
        raise ValueError(
            "no target survived construction; check that the passage lists are "
            "indexed by word_number and that the response file matches them"
        )
    return Corpus(
        P_list, n_list, u_list, f_list, g_list, clusters, wid, FEATURE_NAMES, slots,
        n_context=nctx, max_depth=cfg.max_depth, keys=tkeys
    )


MIN_K_FRAC = 0.2


def min_k_logprob(nll: np.ndarray, frac: float = MIN_K_FRAC) -> float:
    """Min-K% Prob of Shi et al. (2024): mean log-probability of the ``frac``
    least likely tokens.  A passage the reference has memorised has few
    surprising tokens, so its score sits high; the confound table splits Provo
    into tertiles of this score to show whether fitted decay tracks it."""
    nll = np.asarray(nll, dtype=np.float64)
    if nll.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(frac * nll.size)))
    worst = np.sort(nll)[-k:]
    return float(-worst.mean())


def provo_perplexity(provo: ProvoData, scorer) -> dict:
    """Token perplexity and Min-K% of the reference over the Provo passages.

    Printed beside every competence confound's ``delta_hat`` so a reader can see
    that a weaker model is a weaker model and not a zero-decay null, and used
    to split the human fit into contamination tertiles.
    """
    total, count, per, mink = 0.0, 0, {}, {}
    for tid, ws in provo.passages.items():
        nll, n, tokens = scorer.passage_nll([w for w in ws if w])
        total += nll
        count += n
        per[int(tid)] = float(np.exp(nll / n)) if n else float("nan")
        mink[int(tid)] = min_k_logprob(tokens)
    return {
        "model": getattr(scorer, "model_name", "?"),
        "n_passages": len(per),
        "n_tokens": int(count),
        "mean_token_nll": float(total / count) if count else float("nan"),
        "perplexity": float(np.exp(total / count)) if count else float("nan"),
        "min_k_frac": MIN_K_FRAC,
        "per_passage": per,
        "per_passage_min_k": mink,
    }


def gate_g0(corpus: Corpus, provo: ProvoData, sample: int | None = None,
            seed: int = 0) -> dict:
    """G0: the cache is arithmetic on the right object.

    Checks that every cache row is a proper distribution, that no row is
    constant across depths for every target (which would mean the ablation did
    nothing), and that the depth-``K`` row differs from the depth-0 row by more
    than numerical noise.  A pass does not prove the tokenisation is right; a
    failure proves it is wrong, which is what a gate is for.
    """
    # The default reads every target.  A sample of 25 out of some 2,600 passes
    # with probability well over 0.9 when a whole class of targets is degenerate,
    # and the whole-corpus pass is one walk over a 40 MB buffer.
    if sample is None:
        idx = np.arange(len(corpus))
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(corpus), size=min(sample, len(corpus)), replace=False)
    bad_sum, degenerate, spans = 0, 0, []
    for t in idx:
        tg = corpus.target(int(t))
        if np.abs(tg.P.sum(axis=1) - 1.0).max() > 1e-6:
            bad_sum += 1
        span = float(np.abs(tg.P[-1] - tg.P[0]).max())
        spans.append(span)
        if span < 1e-9:
            degenerate += 1
    return {
        "gate": "G0",
        "n_checked": int(len(idx)),
        "rows_not_normalised": int(bad_sum),
        "degenerate_targets": int(degenerate),
        "median_depth_span": float(np.median(spans)) if spans else float("nan"),
        "passed": bad_sum == 0 and degenerate == 0,
    }
