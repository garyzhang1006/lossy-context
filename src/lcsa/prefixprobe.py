"""Pricing the truncated prompt: the same depths rescored from the passage start.

The retention kernel deletes words from the front of the context, so the depth-j
row of the cache conditions on a bare span of ``j`` words that begins
mid-sentence, with no document start and no bos token except at ``j = 0``.  A
reference trained on documents has seen little text of that shape, so part of
the movement in ``D_j`` is the model reacting to an unfamiliar prompt rather
than to anything a reader retained.

This probe prices that reaction.  For a seeded sample of targets and every depth
it scores the frozen candidate set twice: once under the bare span the build
caches, and once under the passage-initial span of the same ``j`` words, which
begins where the passage begins and so carries the document start the truncated
row lacks.  The gap is the total-variation distance between the two candidate
distributions, ``0.5 * sum |p - q|``, the norm gate G2 already applies to the
incremental displacements.

Two depths are structurally exempt and their gap is exactly zero.  At ``j = 0``
both prompts are empty and the scorer falls back to the bos token, and at ``j``
equal to the target's own context length the last ``j`` words *are* the
passage's first ``j`` words.  Every depth between them is a truncated row, and
those are what the probe is about; the summary carries the median over all
probed rows and the median over the truncated ones separately.

The gap bounds the format effect rather than measuring it, because the two
prompts differ in content as well as in shape: no prompt can both begin at the
passage start and condition on the last ``j`` words alone.  A small median gap
is therefore evidence that the truncated format costs little; a large one says
only that the two prompts are far apart, and the identification ceiling of E2
stays the upper bound on what the geometry supports.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from lcsa.build import BuildConfig
from lcsa.cache import context_string
from lcsa.experiments import Artifacts

log = logging.getLogger(__name__)

__all__ = ["DEFAULT_SAMPLE", "DEFAULT_SEED", "DIAGNOSTIC", "PROBE_STEM",
           "prefix_restoration_probe", "run", "sample_targets"]

#: Targets drawn by default.  The probe pays two forwards per (target, depth),
#: so fifty targets is about four thousand contexts on the Provo depth ladder,
#: a few minutes beside the build's own hours.
DEFAULT_SAMPLE = 50
DEFAULT_SEED = 0
PROBE_STEM = "prefix_probe"
#: Name this record claims in the run manifest's diagnostics block.
DIAGNOSTIC = "prefix_restoration_probe"


def sample_targets(keys, sample: int = DEFAULT_SAMPLE, seed: int = DEFAULT_SEED) -> list:
    """Indices of the probed targets: a seeded draw without replacement.

    The indices are returned in ascending order, so the seed fixes *which*
    targets are probed and the scoring order never depends on the order the
    draw happened to produce.
    """
    n = len(keys)
    take = min(int(sample), n)
    if take <= 0:
        raise ValueError(f"the probe needs at least one target; sample={sample} over {n} keys")
    rng = np.random.default_rng(int(seed))
    return sorted(int(i) for i in rng.choice(n, size=take, replace=False))


def _probs(lp: np.ndarray) -> np.ndarray:
    """Candidate log-probabilities to probabilities, as ``target_cache`` does it."""
    p = np.exp(lp - lp.max()) * math.exp(float(lp.max()))
    if not np.all(np.isfinite(p)):
        raise FloatingPointError("non-finite candidate distribution in the prefix probe")
    return p


def _stats(values) -> dict:
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "median_tv_gap": float("nan"), "mean_tv_gap": float("nan"),
                "max_tv_gap": float("nan")}
    return {"n": int(a.size), "median_tv_gap": float(np.median(a)),
            "mean_tv_gap": float(a.mean()), "max_tv_gap": float(a.max())}


def prefix_restoration_probe(
    provo,
    keys,
    candidate_sets,
    scorer,
    sample: int = DEFAULT_SAMPLE,
    seed: int = DEFAULT_SEED,
    max_depth=None,
) -> dict:
    """Total-variation gap between the bare cache and a passage-initial rescoring.

    ``keys`` is ``targets.csv`` as ``(text_id, word_number)`` pairs and
    ``candidate_sets`` is ``candidates.json``, so the probe scores exactly the
    candidate lists the build froze rather than a set reconstructed here.
    """
    if len(keys) != len(candidate_sets):
        raise ValueError(
            f"the probe was given {len(keys)} target keys and {len(candidate_sets)} "
            "candidate lists; both come from the same build directory"
        )
    depth_cap = BuildConfig.max_depth if max_depth is None else int(max_depth)
    chosen = sample_targets(keys, sample, seed)

    rows, per_target = [], []
    for i in chosen:
        tid, wn = int(keys[i][0]), int(keys[i][1])
        passage = provo.passages.get(tid)
        if passage is None:
            raise KeyError(
                f"passage {tid} is absent from the Provo record, so target {(tid, wn)} "
                "cannot be rescored; targets.csv and --provo-dir come from different builds"
            )
        words = [str(w) for w in candidate_sets[i]]
        # The passage's real words in reading order.  Provo drops each passage's
        # first word, so "passage-initial" means the earliest word the corpus
        # carries; that is still a document start as far as the prompt is
        # concerned, which is the property the truncated span lacks.
        head = [w for w in passage if w]
        n_context = sum(1 for w in passage[:wn] if w)
        K = min(n_context, depth_cap)
        gaps = []
        for j in range(K + 1):
            bare = context_string(passage, wn, j)
            restored = " ".join(head[:j])
            p = _probs(scorer.candidate_logprobs(bare, words))
            q = _probs(scorer.candidate_logprobs(restored, words))
            tv = 0.5 * float(np.abs(p - q).sum())
            gaps.append(tv)
            rows.append({"text_id": tid, "word_number": wn, "depth": j, "K": K,
                         "identical_prompt": bare == restored, "tv_gap": tv})
        per_target.append({"text_id": tid, "word_number": wn, "K": K,
                           "median_tv_gap": float(np.median(gaps))})

    truncated = [r["tv_gap"] for r in rows if not r["identical_prompt"]]
    overall = _stats(r["tv_gap"] for r in rows)
    by_depth = []
    for j in sorted({r["depth"] for r in rows}):
        block = [r for r in rows if r["depth"] == j]
        stats = _stats(r["tv_gap"] for r in block)
        by_depth.append({"depth": int(j), "n_targets": stats.pop("n"),
                         "n_identical_prompts": sum(1 for r in block if r["identical_prompt"]),
                         **stats})
    summary = {
        "n_targets_probed": len(chosen),
        "n_targets_available": len(keys),
        "sample": int(sample),
        "seed": int(seed),
        "max_depth": int(depth_cap),
        "n_pairs": len(rows),
        "n_truncated_pairs": len(truncated),
        "median_tv_gap": overall["median_tv_gap"],
        "median_tv_gap_truncated": _stats(truncated)["median_tv_gap"],
        "mean_tv_gap": overall["mean_tv_gap"],
        "max_tv_gap": overall["max_tv_gap"],
        "model": str(getattr(scorer, "model_name", "?")),
    }
    log.info("prefix probe: %d targets, %d contexts rescored, median TV gap %.4g "
             "(%.4g over the truncated depths)", len(chosen), 2 * len(rows),
             summary["median_tv_gap"], summary["median_tv_gap_truncated"])
    return {
        "diagnostic": DIAGNOSTIC,
        "summary": summary,
        "by_depth": by_depth,
        "per_target": per_target,
        # One row per rescored (target, depth), so any aggregate above can be
        # recomputed from the record rather than trusted.
        "pairs": rows,
        "note": "Each depth is scored twice: under the bare span the cache uses, and "
                "under the passage-initial span of the same length, which begins where "
                "the passage begins. Depth 0 and the depth equal to a target's own "
                "context length put the same prompt on both sides and their gap is "
                "exactly zero. The gap bounds the prompt-format effect rather than "
                "measuring it, since the two prompts differ in content as well as in "
                "shape; the E2 ceiling remains the bound on what the geometry supports.",
    }


def run(provo, keys, candidate_sets, scorer, out_dir, sample: int = DEFAULT_SAMPLE,
        seed: int = DEFAULT_SEED, max_depth=None) -> dict:
    """Run the probe and write ``prefix_probe.json`` and ``prefix_probe.csv``.

    The JSON carries ``diagnostic`` and ``summary`` keys, which is what
    :func:`lcsa.manifest.write_manifest` folds into the run manifest.
    """
    rec = prefix_restoration_probe(provo, keys, candidate_sets, scorer, sample=sample,
                                   seed=seed, max_depth=max_depth)
    art = Artifacts(out_dir, PROBE_STEM)
    art.table(PROBE_STEM, rec["by_depth"])
    art.save(PROBE_STEM, rec)
    return rec
