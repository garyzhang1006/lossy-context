import numpy as np
import pandas as pd
import pytest

from lcsa.experiments import e5_participants as e5
from lcsa.kernels import POWER
from lcsa.likelihood import NAIVE
from conftest import draw, make_corpus


def _participants(corpus, theta, n_people=6, seed=3):
    """Each person answers every target once; a dict of 0/1 count lists."""
    rng = np.random.default_rng(seed)
    counts = {}
    for p in range(n_people):
        full = draw(corpus, theta, NAIVE, n_per_target=1, seed=int(rng.integers(1 << 30)))
        counts[f"p{p}"] = [t.n.copy() for t in full]
    return counts


def test_participant_counts_reads_the_frozen_candidate_lists():
    corpus = make_corpus(n_targets=8, n_clusters=2, seed=1)
    keys = [(1, i + 1) for i in range(len(corpus))]
    cands = [[f"w{j}" for j in range(t.V)] for t in corpus]
    rows = []
    for pid in ("a", "b"):
        for (tid, wn), words in zip(keys, cands):
            rows.append({"participant": pid, "text_id": tid, "word_number": wn, "response": words[0]})
    rows.append({"participant": "a", "text_id": 1, "word_number": 1, "response": "not-a-candidate"})
    frame = pd.DataFrame(rows)
    old = e5.MIN_TARGETS
    e5.MIN_TARGETS = 1
    try:
        counts = e5.participant_counts(corpus, keys, cands, frame)
    finally:
        e5.MIN_TARGETS = old
    assert set(counts) == {"a", "b"}
    assert all(c.shape == (t.V,) for c, t in zip(counts["a"], corpus))
    assert sum(c.sum() for c in counts["a"]) == len(corpus)
    with pytest.raises(ValueError):
        e5.participant_counts(corpus, keys[:-1], cands, frame)


def test_fit_and_summarise_run_on_synthetic_participants():
    corpus = make_corpus(n_targets=40, n_clusters=5, seed=2)
    theta = np.array([0.6, 0.3, 0.2])
    counts = _participants(corpus, theta)
    pooled = corpus.with_counts([sum(counts[p][i] for p in counts) for i in range(len(corpus))])
    th = e5.pooled_theta(pooled, [NAIVE], POWER, seed=0)
    assert th["naive"][0] == 0.0
    rows = e5.fit_participants(corpus, counts, th, [NAIVE], None, POWER, seed=0)
    assert len(rows) == len(counts) and not any(r["failed"] for r in rows)
    summ = e5.summarise(rows, external={"p0": 0.4, "p1": 0.5, "p2": 0.1, "p3": 0.9, "p4": 0.2})
    assert summ[0]["n_participants"] == len(counts)
    assert np.isfinite(summ[0]["median_delta_pinned"])
    assert summ[0]["n_external_shared"] == 5
