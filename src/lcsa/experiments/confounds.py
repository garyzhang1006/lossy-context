"""The competence confounds: weaker references reading full context.

A language model reading the whole context is not a zero-decay reader, since
full input access is not zero functional retention, so these fits are labelled
confounds and never nulls.  Each row carries the reference's Provo token
perplexity so that a reader can see competence and fitted decay side by side
and the paper cannot treat a weak model as a floor.
"""

from __future__ import annotations

import logging


from lcsa.experiments import Artifacts
from lcsa.experiments.e3_nulls import fit_and_profile
from lcsa.kernels import POWER

log = logging.getLogger(__name__)

__all__ = ["run"]


def run(references: dict, models, out_dir, kernel=POWER, seed: int = 0) -> list[dict]:
    """One fit per (reference, estimator), written to ``confounds.csv``.

    ``references`` maps a name to ``(corpus, perplexity_record)``; the record
    is the ``perplexity.json`` the build wrote, or ``None`` when it is missing,
    in which case the row says so rather than carrying a made-up number.
    """
    art = Artifacts(out_dir, "confounds")
    rows = []
    for name, (corpus, ppl) in references.items():
        for m in models:
            row = fit_and_profile(corpus, m, kernel, seed=seed)
            rows.append({
                "reader": name,
                "kind": "competence confound",
                "perplexity": float(ppl["perplexity"]) if ppl else float("nan"),
                "n_tokens": int(ppl["n_tokens"]) if ppl else 0,
                **row,
            })
            log.info("%s under %s: delta %.4f, perplexity %s", name, m.name,
                     row["delta_hat"], rows[-1]["perplexity"])
    art.table("confounds", rows)
    art.save("confounds", {"note": "full-context readers are not zero-decay nulls; "
                                   "these rows are labelled confounds and their fitted "
                                   "decay is not interpretable as estimator failure",
                           "rows": rows})
    return rows
