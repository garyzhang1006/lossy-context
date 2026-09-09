"""G3, the day-2 reliability gate, from the built cache and the eye-tracking arm.

The cloze phenotype is the per-context response count vector, compared with
itself across participant halves and with the full-context reference, both
jackknife-debiased so that the ratio the gate consumes is not inflated by the
plug-in bias the registration works out.  The gaze phenotype is per-word first-
pass gaze duration split across halves of the participant set.
"""

from __future__ import annotations

import logging

from lcsa.experiments import Artifacts
from lcsa.gates import g3_reliability
from lcsa.reliability import js_reliability, split_half_reliability

log = logging.getLogger(__name__)

__all__ = ["run"]


def run(corpus, provo, out_dir, seed: int = 0, n_splits: int = 20, n_sim: int = 20) -> dict:
    counts = [t.n for t in corpus]
    refs = [t.P[-1] for t in corpus]  # row K is the full context
    js = js_reliability(counts, refs, n_splits=n_splits, n_sim=n_sim, seed=seed)
    gaze = float("nan")
    n_participants = 0
    if provo is not None and getattr(provo, "gaze", None) is not None:
        gz = provo.gaze.dropna(subset=["gaze"])
        items = gz["text_id"].astype(int).to_numpy() * 10_000 + gz["word_number"].astype(int).to_numpy()
        gaze = split_half_reliability(gz["gaze"].to_numpy(), items, seed=seed,
                                      participants=gz["participant_id"].to_numpy())
        n_participants = int(gz["participant_id"].nunique())
    else:
        log.warning("no eye-tracking arm: G3 is evaluated on the cloze phenotype alone")
    gate = g3_reliability(js, gaze)
    rec = {"js": js, "gaze_split_half": gaze, "n_participants": n_participants,
           "n_clusters": int(corpus.n_clusters), "g3": gate}
    Artifacts(out_dir, "reliability").save("reliability", rec)
    log.info("G3 %s: JS ratio %.3f, gaze split-half %.3f", "passed" if gate.passed else "FAILED",
             gate.measured["js_ratio"], gaze)
    return rec
