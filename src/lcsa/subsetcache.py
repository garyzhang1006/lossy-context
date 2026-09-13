"""Pricing every subset of a short context, not only its suffixes.

The build caches ``K + 1`` rows per target, one per retained depth, because
graded truncation puts all its mass on the contiguous suffixes.  Independent
deletion does not: it keeps each of the ``K`` positions with its own
probability, so its marginal runs over all ``2^K`` retention patterns and most
of them are contexts with a hole in the middle that the nested cache never
holds.  Prediction 10 compares the two kernels on the same targets, and it
cannot be scored from the nested cache at all.

This module is the producer of the missing object.  For every target whose
depth is at or below ``K_MAX`` it scores the frozen candidate set under each of
the ``2^K`` masks and returns a ``(2^K, V)`` block in bitmask order, using the
same log-probability-to-probability conversion
:meth:`lcsa.cache.ReferenceScorer.target_cache` uses, so a row of one cache and
a row of the other are the same kind of number.

Bit ``j - 1`` of a mask means the word at distance ``j`` is retained, distance 1
being the word immediately before the target, which is the convention
:func:`lcsa.experiments.e1_exactness.subset_masks` states and
:func:`~lcsa.experiments.e1_exactness.graded_marginal_from_subset` relies on:
it reads the suffix masks ``(1 << k) - 1`` and treats them as the depth-``k``
rows of the nested cache.  Those rows are therefore built from exactly the
string :func:`lcsa.cache.context_string` gives at depth ``k``, and a test
asserts the two prompts are the same characters rather than the same idea.

The cap is what makes this affordable.  The cost is ``sum(2^K)`` forward passes
over the selected targets, so a target at ``K = 8`` costs 256 contexts where the
nested cache costs 9, and one at ``K = 20`` would cost a million.  ``K_MAX``
defaults to 8, which on Provo selects the targets near the start of a passage
and leaves the bridging comparison to the short contexts the registration
already scopes it to.

Nothing here declares a manifest diagnostic.  The convention in
:mod:`lcsa.manifest` collects a ``summary`` block that names a headline number,
and this artifact names none: it is an input to the bridging report, whose
numbers reach the scorecard through ``e1_summary.json``.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from lcsa.build import BuildConfig
from lcsa.cache import context_string
from lcsa.experiments import Artifacts
from lcsa.store import save_sub_caches

log = logging.getLogger(__name__)

__all__ = ["K_MAX", "BRIDGING_TARGETS", "STEM", "mask_context", "selected_targets",
           "subset_cache", "subset_caches", "run"]

#: Depth at or below which a target is cheap enough to enumerate.  The producer
#: pays ``2^K`` forwards for a target the build paid ``K + 1`` for, so this is
#: the only knob between a few thousand contexts and a few million.
K_MAX = 8
STEM = "sub_caches"

#: How many Provo positions the cap admits, frozen with the rest of the design.
#: The paper prints this number as a checkable property of the corpus, so the
#: producer refuses a build whose selection disagrees with it rather than
#: enumerating a different set of targets under the registered name.
BRIDGING_TARGETS = 439


def _probs(lp: np.ndarray, key) -> np.ndarray:
    """Candidate log-probabilities to probabilities, as ``target_cache`` does it."""
    p = np.exp(lp - lp.max()) * math.exp(float(lp.max()))
    if not np.all(np.isfinite(p)):
        raise FloatingPointError(
            f"non-finite subset-cache row for target {tuple(int(x) for x in key)}"
        )
    return p


def mask_context(passage, word_number: int, K: int, mask: int) -> str:
    """The prompt retaining the positions ``mask`` names, in reading order.

    ``passage`` is indexed by Provo word_number and carries empty slots for the
    numbers Provo does not have, so the distances count real words, exactly as
    :func:`lcsa.cache.context_string` counts them.  Every word further back than
    distance ``K`` is dropped whatever the mask says, which is what makes the
    suffix mask ``(1 << k) - 1`` the depth-``k`` context and not a longer one.
    """
    prefix = [w for w in passage[:word_number] if w]
    n = len(prefix)
    keep = sorted(n - j for j in range(1, K + 1) if (mask >> (j - 1)) & 1)
    return " ".join(prefix[i] for i in keep)


def selected_targets(provo, keys, k_max: int = K_MAX, max_depth=None) -> list[dict]:
    """The targets the cap admits, with the ``K`` each of them will be enumerated at.

    ``K`` is computed the way :func:`lcsa.build.build_corpus` computes it, from
    the count of real words below the target capped at the build's own depth
    cap, so a target selected here has the same ``K`` as its row of the nested
    cache and the two are comparable position for position.
    """
    depth_cap = BuildConfig.max_depth if max_depth is None else int(max_depth)
    out = []
    for i, (tid, wn) in enumerate(keys):
        tid, wn = int(tid), int(wn)
        passage = provo.passages.get(tid)
        if passage is None:
            raise KeyError(
                f"passage {tid} is absent from the Provo record, so target {(tid, wn)} "
                "cannot be enumerated; targets.csv and --provo-dir come from different builds"
            )
        n_context = sum(1 for w in passage[:wn] if w)
        K = min(n_context, depth_cap)
        if K <= int(k_max):
            out.append({"index": i, "text_id": tid, "word_number": wn, "K": int(K),
                        "n_context": int(n_context), "n_contexts_scored": 1 << int(K)})
    return out


def subset_cache(passage, word_number: int, words, scorer, K: int, key=None) -> np.ndarray:
    """The ``(2^K, V)`` block for one target, row ``m`` scored under mask ``m``."""
    key = key if key is not None else (-1, int(word_number))
    rows = [_probs(scorer.candidate_logprobs(mask_context(passage, word_number, K, m), words), key)
            for m in range(1 << int(K))]
    P = np.asarray(rows, dtype=np.float64)
    if P.shape != (1 << int(K), len(words)):
        raise ValueError(
            f"target {tuple(int(x) for x in key)}: subset cache has shape {P.shape}, "
            f"expected {(1 << int(K), len(words))}"
        )
    return P


def subset_caches(provo, keys, candidate_sets, scorer, k_max: int = K_MAX,
                  max_depth=None) -> dict[tuple[int, int], np.ndarray]:
    """Every selected target's subset cache, keyed by ``(text_id, word_number)``.

    ``keys`` is ``targets.csv`` as ``(text_id, word_number)`` pairs and
    ``candidate_sets`` is ``candidates.json``, so this scores exactly the
    candidate lists the build froze rather than a set reconstructed here.
    """
    if len(keys) != len(candidate_sets):
        raise ValueError(
            f"the subset cache was given {len(keys)} target keys and "
            f"{len(candidate_sets)} candidate lists; both come from the same build directory"
        )
    chosen = selected_targets(provo, keys, k_max=k_max, max_depth=max_depth)
    n_ctx = sum(r["n_contexts_scored"] for r in chosen)
    log.info("subset cache: %d of %d targets at K <= %d, %d contexts to score",
             len(chosen), len(keys), int(k_max), n_ctx)
    out: dict[tuple[int, int], np.ndarray] = {}
    for r in chosen:
        passage = provo.passages[r["text_id"]]
        key = (r["text_id"], r["word_number"])
        words = [str(w) for w in candidate_sets[r["index"]]]
        out[key] = subset_cache(passage, r["word_number"], words, scorer, r["K"], key=key)
    log.info("subset cache: %d targets enumerated, %d contexts scored", len(out), n_ctx)
    return out


def run(provo, keys, candidate_sets, scorer, out_dir, k_max: int = K_MAX,
        max_depth=None, expect_n: int | None = None) -> dict:
    """Enumerate and write ``sub_caches.npz`` beside its selection record.

    The ``.npz`` is the artifact ``lcsa e1 --sub-caches`` reads; the JSON and CSV
    beside it say which targets were selected and what each one cost, so the
    price of the leg is readable without opening the arrays.

    ``expect_n`` is the registered target count, and the real Provo run passes
    :data:`BRIDGING_TARGETS` so that a corpus or a depth cap that no longer
    selects the registered positions fails here instead of reaching the paper.
    Leaving it ``None`` enumerates whatever the cap admits.
    """
    caches = subset_caches(provo, keys, candidate_sets, scorer, k_max=k_max,
                           max_depth=max_depth)
    if not caches:
        raise ValueError(
            f"no target of the {len(keys)} in targets.csv has K <= {int(k_max)}, so there "
            "is nothing to enumerate; raise --k-max, at 2^K forwards per target"
        )
    rows = selected_targets(provo, keys, k_max=k_max, max_depth=max_depth)
    if expect_n is not None and len(rows) != int(expect_n):
        raise ValueError(
            f"the cap at K <= {int(k_max)} selects {len(rows)} of the {len(keys)} targets, "
            f"but the registration froze {int(expect_n)}; the corpus, the depth cap or the "
            "target list has changed, so re-freeze the registration before enumerating"
        )
    art = Artifacts(out_dir, STEM)
    path = save_sub_caches(art.dir / f"{STEM}.npz", caches)
    rec = {
        "summary": {
            "n_targets_selected": len(caches),
            "n_targets_available": len(keys),
            "n_targets_registered": None if expect_n is None else int(expect_n),
            "k_max": int(k_max),
            "max_depth": BuildConfig.max_depth if max_depth is None else int(max_depth),
            "max_K_selected": max(r["K"] for r in rows),
            "n_contexts_scored": sum(r["n_contexts_scored"] for r in rows),
            "artifact": path.name,
            "model": str(getattr(scorer, "model_name", "?")),
        },
        "per_target": rows,
        "note": "Row m of a target's block is the candidate distribution under the "
                "retention mask m, bit j-1 meaning the word at distance j is kept. The "
                "suffix masks (1 << k) - 1 are the depth-k rows of the nested cache, "
                "scored from the same prompt string, which is what lets the bridging "
                "report put graded truncation and independent deletion on one object.",
    }
    art.table(STEM, rows)
    art.save(STEM, rec)
    return rec
