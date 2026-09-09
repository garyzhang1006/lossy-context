"""The competence confounds: weaker references reading full context.

A language model reading the whole context is not a zero-decay reader, since
full input access is not zero functional retention, so these fits are labelled
confounds and never nulls.  Each row carries the reference's Provo token
perplexity so that a reader can see competence and fitted decay side by side
and the paper cannot treat a weak model as a floor.

The same table carries the contamination check: the human counts refitted
within tertiles of the primary reference's Min-K% score per passage.  If the
fitted decay tracked memorisation of Provo, the tertile rows would drift with
the score.
"""

from __future__ import annotations

import logging

import numpy as np

from lcsa.experiments import Artifacts
from lcsa.experiments.e3_nulls import fit_and_profile
from lcsa.kernels import POWER

log = logging.getLogger(__name__)

__all__ = ["run", "min_k_tertiles"]

MIN_CLUSTERS_PER_TERTILE = 2


def min_k_tertiles(corpus, ppl: dict) -> list[tuple[int, list[int], float, float]]:
    """Cluster indices of each tertile of the per-passage Min-K% score.

    Returns ``(tertile, cluster_index_list, low, high)`` triples in ascending
    order of the score, or an empty list when the record carries no per-passage
    scores or a tertile would hold fewer than two passages, which is too few
    for a cluster-robust variance.
    """
    scores = ppl.get("per_passage_min_k") or {}
    vals = []
    for i, cid in enumerate(corpus.cluster_ids):
        v = scores.get(str(int(cid)), scores.get(int(cid)))
        if v is None or not np.isfinite(v):
            return []
        vals.append((float(v), i))
    if not vals:
        return []
    vals.sort()
    n = len(vals)
    bounds = [0, n // 3, (2 * n) // 3, n]
    out = []
    for t in range(3):
        block = vals[bounds[t]:bounds[t + 1]]
        if len(block) < MIN_CLUSTERS_PER_TERTILE:
            return []
        out.append((t + 1, [i for _, i in block], block[0][0], block[-1][0]))
    return out


def run(references: dict, models, out_dir, kernel=POWER, seed: int = 0,
        primary: tuple | None = None) -> list[dict]:
    """One fit per (reference, estimator) plus the Min-K% tertile rows, to ``confounds.csv``.

    ``references`` maps a name to ``(corpus, perplexity_record)``; the record
    is the ``perplexity.json`` the build wrote, or ``None`` when it is missing,
    in which case the row says so rather than carrying a made-up number.
    ``primary`` is the same pair for the primary build and drives the tertiles.
    """
    art = Artifacts(out_dir, "confounds")
    rows = []
    for name, (corpus, ppl) in references.items():
        for m in models:
            row = fit_and_profile(corpus, m, kernel, seed=seed)
            rows.append({
                "reader": name,
                "kind": "competence confound",
                "tertile": 0,
                "perplexity": float(ppl["perplexity"]) if ppl else float("nan"),
                "n_tokens": int(ppl["n_tokens"]) if ppl else 0,
                "min_k_low": float("nan"),
                "min_k_high": float("nan"),
                "n_clusters": int(corpus.n_clusters),
                **row,
            })
            log.info("%s under %s: delta %.4f, perplexity %s", name, m.name,
                     row["delta_hat"], rows[-1]["perplexity"])
    tertiles = min_k_tertiles(primary[0], primary[1]) if primary and primary[1] else []
    if primary and not tertiles:
        log.warning("no Min-K%% tertile rows: the primary perplexity.json has no usable "
                    "per-passage scores or a tertile would hold fewer than %d passages",
                    MIN_CLUSTERS_PER_TERTILE)
    for t, idx, lo, hi in tertiles:
        sub = primary[0].subset_clusters(idx)
        for m in models:
            row = fit_and_profile(sub, m, kernel, seed=seed)
            rows.append({
                "reader": "human",
                "kind": "min-k tertile",
                "tertile": int(t),
                "perplexity": float(primary[1]["perplexity"]),
                "n_tokens": int(primary[1]["n_tokens"]),
                "min_k_low": float(lo),
                "min_k_high": float(hi),
                "n_clusters": int(sub.n_clusters),
                **row,
            })
    art.table("confounds", rows)
    art.save("confounds", {"note": "full-context readers are not zero-decay nulls; "
                                   "these rows are labelled confounds and their fitted "
                                   "decay is not interpretable as estimator failure. "
                                   "Tertile rows refit the human counts within tertiles "
                                   "of the primary reference's Min-K% score per passage.",
                           "rows": rows})
    return rows
