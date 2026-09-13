"""E4: the context-limitation sweep, the reading-time diagnostic and the baselines.

The sweep is the measurement the field already trusts, replicated here across
references that have no context decay in them at all.  If a zero-decay reference
still selects a short window, the sweep is measuring something about the
reference rather than about memory, and prediction 8 says that is what happens.
The reading-time leg asks the same question of the fitted kernel: if a null's
spuriously fitted kernel predicts gaze durations nearly as well as the human
one, then reading-time fit is not the external check it is usually taken for.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from lcsa.baselines import (context_slopes, decorativeness_check, hard_window_sweep,
                            passage_slopes, sweep_disagreement)
from lcsa.corpusdata import Corpus
from lcsa.fitting import fit_constrained
from lcsa.inference import score_test
from lcsa.kernels import POWER
from lcsa.likelihood import Model
from lcsa.readers import reader_n0
from lcsa.readingtime import RTResult, _fit_ll, _gauss_ll, reading_time_gain, spillover
from lcsa.experiments import Artifacts
from lcsa.experiments.shards import denull, read_shards, write_shard

log = logging.getLogger(__name__)

__all__ = ["K_GRID", "gaze_table", "window_surprisal", "heldout_delta_ll", "sweep_reference",
           "argmax_picks", "summarise_argmax", "argmax_bootstrap", "rt_gain_table",
           "decorativeness", "run_sweep", "load_stage", "run_argmax_shard", "assemble",
           "merge", "run"]

#: The context-limitation grid, in words of preceding context.
K_GRID = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32)


def gaze_table(provo, keys, unigrams=None, measure: str = "gaze"):
    """Align the eye-tracking arm to the built targets.

    Returns the mean gaze duration per target, the control matrix (log unigram
    frequency and word length), the passage label and the word number, in the
    corpus's own target order.  The word number is what the spillover lag is
    taken by: a target the build dropped leaves a hole in the passage, and a
    lag over the retained rows alone would hand its successor the surprisal of
    the word before the hole.  A target with no eye-tracking record becomes ``nan`` and is dropped
    downstream rather than imputed, because imputing a reading time is inventing
    the measurement the whole leg rests on.
    """
    import pandas as pd

    if provo.gaze is None:
        raise ValueError(
            "this Provo load has no eye-tracking arm; pass require_eye=True to "
            "load_provo, or run the estimation legs only"
        )
    gz = provo.gaze
    col = measure if measure in gz.columns else "gaze"
    agg = (gz.groupby(["text_id", "word_number"])[col]
             .mean().rename("y").reset_index())
    lut = {(int(a), int(b)): float(c) for a, b, c in
           zip(agg["text_id"], agg["word_number"], agg["y"])}
    wmap = {(int(r.text_id), int(r.word_number)): str(r.word)
            for r in provo.words.itertuples()}
    y, ctrl, passage, position = [], [], [], []
    for tid, wn in keys:
        y.append(lut.get((int(tid), int(wn)), np.nan))
        w = wmap.get((int(tid), int(wn)), "")
        lf = float(unigrams(w)) if unigrams is not None else 0.0
        ctrl.append([lf, float(len(w))])
        passage.append(int(tid))
        position.append(int(wn))
    return (np.asarray(y, dtype=np.float64),
            np.asarray(ctrl, dtype=np.float64),
            np.asarray(passage),
            np.asarray(position))


def window_surprisal(corpus: Corpus, k: int, P_override=None) -> np.ndarray:
    """``-log p_ref(w_t | last k words)`` for each target's own word.

    ``P_override`` supplies an alternative reference: a list of ablation blocks
    in the corpus's target order, which is how the tilted zero-decay references
    enter the sweep at no marginal GPU cost, or a whole :class:`Corpus` built
    under another checkpoint on the same frozen candidate sets.
    """
    if isinstance(P_override, Corpus):
        if len(P_override) != len(corpus):
            raise ValueError(
                f"the reference cache has {len(P_override)} targets but the primary has "
                f"{len(corpus)}; references must be built on the primary's targets.csv")
        return window_surprisal(P_override, k)
    out = np.full(len(corpus), np.nan, dtype=np.float64)
    for t, tgt in enumerate(corpus):
        i = int(tgt.target_slot)
        if i < 0 or i >= tgt.V:
            continue
        P = tgt.P if P_override is None else P_override[t]
        j = min(int(k), P.shape[0] - 1)
        p = float(P[j][i])
        out[t] = -np.log(max(p, 1e-300))
    return out


def heldout_delta_ll(
    y: np.ndarray,
    controls: np.ndarray,
    predictor: np.ndarray,
    passage: np.ndarray,
    n_folds: int | None = None,
    use_mixed: bool = True,
    seed: int = 0,
    position: np.ndarray | None = None,
) -> float:
    """Held-out log-likelihood gain from adding one predictor and its spillover.

    ``position`` is the word number of each row inside its passage; with it the
    spillover is lagged by word number rather than by row, see ``spillover``.

    ``n_folds=None`` means leave one passage out, which is what the registered
    selection rule specifies; folds partition passages so no passage is ever in
    both arms.
    """
    y = np.asarray(y, dtype=np.float64)
    passage = np.asarray(passage)
    ctrl = np.atleast_2d(np.asarray(controls, dtype=np.float64))
    if ctrl.shape[0] != y.size:
        ctrl = ctrl.T
    base = np.column_stack([np.ones(y.size), ctrl])
    x = np.asarray(predictor, dtype=np.float64)
    full = np.column_stack([base, x, spillover(x, passage, position=position)])
    ok = np.isfinite(y) & np.isfinite(full).all(axis=1)
    y, base, full, passage = y[ok], base[ok], full[ok], passage[ok]
    if y.size < 20:
        return float("nan")
    uniq = np.unique(passage)
    if n_folds is None or n_folds >= uniq.size:
        folds = [np.array([u]) for u in uniq]
    else:
        rng = np.random.default_rng(seed)
        folds = np.array_split(rng.permutation(uniq), n_folds)
    tot = 0.0
    for f in folds:
        te = np.isin(passage, f)
        tr = ~te
        if tr.sum() < full.shape[1] + 3 or te.sum() == 0:
            continue
        bb, sb, _, _ = _fit_ll(y[tr], base[tr], passage[tr], use_mixed)
        bf, sf, _, _ = _fit_ll(y[tr], full[tr], passage[tr], use_mixed)
        tot += _gauss_ll(y[te], full[te], bf, sf) - _gauss_ll(y[te], base[te], bb, sb)
    return float(tot / y.size)


def sweep_reference(
    corpus: Corpus,
    gaze: np.ndarray,
    controls: np.ndarray,
    passage: np.ndarray,
    name: str,
    P_override=None,
    k_grid=K_GRID,
    n_folds: int | None = None,
    use_mixed: bool = True,
    seed: int = 0,
    position: np.ndarray | None = None,
) -> dict:
    """The whole delta-log-likelihood curve for one reference, plus its argmax."""
    curve = []
    for k in k_grid:
        s = window_surprisal(corpus, int(k), P_override)
        curve.append({"k": int(k),
                      "delta_ll": heldout_delta_ll(gaze, controls, s, passage,
                                                   n_folds, use_mixed, seed,
                                                   position=position)})
    fin = [c for c in curve if np.isfinite(c["delta_ll"])]
    best = max(fin, key=lambda c: c["delta_ll"])["k"] if fin else None
    return {"reference": name, "curve": curve, "argmax_k": best,
            "k_grid": list(int(k) for k in k_grid)}


def argmax_picks(
    corpus: Corpus,
    gaze: np.ndarray,
    controls: np.ndarray,
    passage: np.ndarray,
    P_override=None,
    k_grid=K_GRID,
    reps=range(200),
    n_folds: int = 5,
    use_mixed: bool = False,
    seed: int = 0,
    position: np.ndarray | None = None,
) -> list[dict]:
    """The selected window on each passage resample ``b`` in ``reps``.

    Replicate ``b`` resamples passages from ``seed + b + 1``, so a shard of
    replicates is the same numbers whether or not the others ran alongside it.
    A replicate on which no grid point gives a finite gain records ``None``.
    """
    passage = np.asarray(passage)
    position = None if position is None else np.asarray(position)
    uniq = np.unique(passage)
    surp = {int(k): window_surprisal(corpus, int(k), P_override) for k in k_grid}
    rows = []
    for b in reps:
        rng = np.random.default_rng(seed + b + 1)
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([np.flatnonzero(passage == p) for p in pick])
        best_k, best_v = None, -np.inf
        for k in k_grid:
            v = heldout_delta_ll(gaze[idx], np.asarray(controls)[idx],
                                 surp[int(k)][idx], passage[idx], n_folds,
                                 use_mixed, seed,
                                 position=None if position is None else position[idx])
            if np.isfinite(v) and v > best_v:
                best_k, best_v = int(k), v
        rows.append({"replicate": int(b), "best_k": best_k})
    return rows


def summarise_argmax(rows: list[dict]) -> dict:
    """Distribution of the selected window over replicate rows.

    A replicate with no finite gain carries ``None`` in memory and ``nan`` after
    a shard round trip; both mean the same unusable replicate.
    """
    picks = [int(r["best_k"]) for r in rows
             if r.get("best_k") is not None and np.isfinite(r["best_k"])]
    if not picks:
        return {"n_boot": int(len(rows)), "n_usable": 0, "distribution": {}}
    vals, counts = np.unique(np.asarray(picks), return_counts=True)
    return {
        "n_boot": int(len(rows)),
        "n_usable": len(picks),
        "distribution": {int(v): int(c) for v, c in zip(vals, counts)},
        "mode_k": int(vals[int(np.argmax(counts))]),
        "median_k": float(np.median(picks)),
    }


def argmax_bootstrap(
    corpus: Corpus,
    gaze: np.ndarray,
    controls: np.ndarray,
    passage: np.ndarray,
    P_override=None,
    k_grid=K_GRID,
    n_boot: int = 200,
    n_folds: int = 5,
    use_mixed: bool = False,
    seed: int = 0,
    position: np.ndarray | None = None,
) -> dict:
    """Cluster bootstrap of the selected window.

    An argmax over a discrete grid is lumpy under resampling, so the honest
    summary is the distribution over grid points rather than a point estimate
    with an interval drawn around it.
    """
    return summarise_argmax(argmax_picks(corpus, gaze, controls, passage, P_override,
                                         k_grid, range(n_boot), n_folds, use_mixed, seed,
                                         position=position))


def rt_gain_table(
    corpus: Corpus,
    gaze: np.ndarray,
    controls: np.ndarray,
    passage: np.ndarray,
    fitted: dict[str, tuple[np.ndarray, Model]],
    kernel=POWER,
    n_folds: int = 5,
    use_mixed: bool = True,
    seed: int = 0,
    position: np.ndarray | None = None,
) -> list[dict]:
    """Held-out gain of each fitted kernel's surprisal over full-context surprisal.

    ``fitted`` maps a reader's name to its fitted parameter vector and estimator.
    Prediction 7 is scored from the ratio of a null's gain to the human gain, and
    a high ratio means reading-time fit cannot tell a spurious kernel from a real
    one.
    """
    from lcsa.readingtime import surprisal_from_corpus

    full = window_surprisal(corpus, 10 ** 6)
    rows = []
    for name, (theta, model) in fitted.items():
        lossy = surprisal_from_corpus(corpus, theta, model, kernel)
        r: RTResult = reading_time_gain(gaze, full, lossy, controls, passage,
                                        n_folds=n_folds, use_mixed=use_mixed, seed=seed,
                                        position=position)
        rows.append({
            "reader": name,
            "estimator": model.name,
            "delta": float(theta[0]),
            "gain_per_word": r.gain_per_word,
            "baseline_ll": r.baseline_ll,
            "full_ll": r.full_ll,
            "n_words": r.n_words,
            "estimator_backend": r.estimator,
            "converged": r.converged,
            "note": r.note,
        })
    human = next((r["gain_per_word"] for r in rows if r["reader"] == "human"), None)
    for r in rows:
        r["fraction_of_human_gain"] = (
            float(r["gain_per_word"] / human)
            if human not in (None, 0.0) and np.isfinite(r["gain_per_word"]) else float("nan")
        )
    return rows


def _passage_statistic(corpus: Corpus, model: Model, kernel=POWER, seed: int = 0,
                       null_fit=None) -> np.ndarray:
    """Each passage's share of the efficient score, in cluster-robust units."""
    st = score_test(corpus, model, kernel, null_fit=null_fit, n_starts=2, seed=seed,
                    n_wild=0)
    sd = float(np.sqrt(st.var_cr1))
    return st.s_cluster / sd if sd > 0 else st.s_cluster


def decorativeness(corpus: Corpus, model: Model, kernel=POWER, seed: int = 0) -> dict:
    """The two model-free slopes against the statistic, on the same two readers.

    The null reader is E3's plain floor: responses redrawn at each target's own
    count from the human fit with ``delta`` pinned at zero, so the readers
    differ in decay and in nothing else.  Everything is then per passage, which
    is the unit all three scores are clustered on anyway.
    """
    null_fit = fit_constrained(corpus, model, kernel, n_starts=2, seed=seed)
    null = reader_n0(corpus, null_fit.theta, model, seed=seed, kernel=kernel)
    human_slopes, null_slopes = passage_slopes(corpus), passage_slopes(null)
    return decorativeness_check(
        human_slopes["entropy"], null_slopes["entropy"],
        human_slopes["top1"], null_slopes["top1"],
        _passage_statistic(corpus, model, kernel, seed, null_fit),
        _passage_statistic(null, model, kernel, seed),
    )


STAGE_JSON = "e4_stage.json"


def run_sweep(
    corpus: Corpus,
    gaze: np.ndarray,
    controls: np.ndarray,
    passage: np.ndarray,
    models,
    out_dir,
    references: dict | None = None,
    fitted: dict | None = None,
    k_grid=K_GRID,
    n_folds: int | None = None,
    use_mixed: bool = True,
    kernel=POWER,
    seed: int = 0,
    position: np.ndarray | None = None,
) -> dict:
    """Stage one of E4: every deterministic table, saved so the shards can skip it.

    The sweep, the reading-time gains and the hard-window likelihoods have no
    replicate loop, so they run once here and the argmax shards only resample.
    """
    art = Artifacts(out_dir, "e4")
    refs = {"primary": None} if not references else references
    sweeps, flat = {}, []
    for name, override in refs.items():
        sw = sweep_reference(corpus, gaze, controls, passage, name, override, k_grid,
                             n_folds, use_mixed, seed, position=position)
        sweeps[name] = sw
        for c in sw["curve"]:
            flat.append({"reference": name, **c})
    art.table("e4_sweep_curves", flat)
    stage = {"references": list(refs), "sweeps": sweeps,
             "n_targets": int(len(gaze)),
             "n_targets_with_gaze": int(np.isfinite(np.asarray(gaze, dtype=float)).sum()),
             # Rows whose spillover lag has no built target (the word after a
             # passage's first word, or after any word below min_responses) drop
             # out of every fit above, and this is the count that survives.
             "n_targets_usable": int((np.isfinite(np.asarray(gaze, dtype=float))
                                      & np.isfinite(spillover(np.ones(len(gaze)), passage,
                                                              position=position))).sum()),
             "model_free": context_slopes(corpus)}
    # The registered falsifier needs two clusters to separate and a response in
    # them; a corpus with neither leaves the row absent rather than asserted.
    if corpus.n_clusters >= 2 and corpus.total_responses > 0:
        stage["decorativeness"] = decorativeness(corpus, models[0], kernel, seed)
    if fitted:
        rows = rt_gain_table(corpus, gaze, controls, passage, fitted, kernel=kernel,
                             n_folds=5, use_mixed=use_mixed, seed=seed, position=position)
        art.table("e4_rt_gain", rows)
        stage["rt_gain"] = rows
    hw = hard_window_sweep(corpus, models[0], windows=tuple(k_grid), n_folds=5, seed=seed)
    art.table("e4_hard_window_likelihood", hw)
    stage["hard_window_likelihood"] = [h.__dict__ for h in hw]
    stage["hard_window_agreement"] = sweep_disagreement({"cloze": hw})
    art.save("e4_stage", stage)
    return stage


def load_stage(out_dir) -> dict:
    p = Path(out_dir) / STAGE_JSON
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; run `lcsa e4 --stage sweep` first")
    return denull(json.loads(p.read_text()))


def run_argmax_shard(
    corpus: Corpus,
    gaze: np.ndarray,
    controls: np.ndarray,
    passage: np.ndarray,
    out_dir,
    reps: range,
    references: dict | None = None,
    k_grid=K_GRID,
    seed: int = 0,
    position: np.ndarray | None = None,
) -> list[dict]:
    """Stage two of E4: argmax replicates ``reps`` for every reference."""
    refs = {"primary": None} if not references else references
    rows = []
    for name, override in refs.items():
        for r in argmax_picks(corpus, gaze, controls, passage, override, k_grid, reps,
                              seed=seed, position=position):
            rows.append({"reference": name, **r})
    write_shard(out_dir, "e4_argmax", reps, rows)
    return rows


def assemble(stage: dict, boot_rows: list[dict], out_dir) -> dict:
    """Stage three of E4: the summary from the sweep stage and the argmax rows."""
    art = Artifacts(out_dir, "e4")
    by_ref: dict[str, list[dict]] = {name: [] for name in stage["references"]}
    for r in boot_rows:
        by_ref.setdefault(r["reference"], []).append(r)
    boots = {name: summarise_argmax(rows) for name, rows in by_ref.items()}
    sweeps = stage["sweeps"]
    res = {
        "sweeps": sweeps,
        "argmax_bootstrap": boots,
        "selected": {n: sw["argmax_k"] for n, sw in sweeps.items()},
        "prediction_8": _prediction_8(sweeps),
        "model_free": stage["model_free"],
    }
    if "rt_gain" in stage:
        res["rt_gain"] = stage["rt_gain"]
    res["hard_window_likelihood"] = stage["hard_window_likelihood"]
    res["hard_window_agreement"] = stage["hard_window_agreement"]
    if "decorativeness" in stage:
        res["decorativeness"] = stage["decorativeness"]
    art.save("e4_summary", res)
    return res


def merge(out_dir, n_boot: int | None = None) -> dict:
    """Combine the sweep stage and the argmax shards on disk."""
    return assemble(load_stage(out_dir), read_shards(out_dir, "e4_argmax", n_boot), out_dir)


def run(
    corpus: Corpus,
    gaze: np.ndarray,
    controls: np.ndarray,
    passage: np.ndarray,
    models,
    out_dir,
    references: dict | None = None,
    fitted: dict | None = None,
    k_grid=K_GRID,
    n_boot: int = 200,
    n_folds: int | None = None,
    use_mixed: bool = True,
    kernel=POWER,
    seed: int = 0,
    position: np.ndarray | None = None,
) -> dict:
    """Full E4 leg: the sweep across references, the argmax bootstrap, the gains."""
    stage = run_sweep(corpus, gaze, controls, passage, models, out_dir, references,
                      fitted, k_grid, n_folds, use_mixed, kernel, seed, position=position)
    rows = run_argmax_shard(corpus, gaze, controls, passage, out_dir, range(n_boot),
                            references, k_grid, seed, position=position)
    return assemble(stage, rows, out_dir)


def _prediction_8(sweeps: dict) -> dict:
    """Does the sweep disagree across zero-decay references, or bottom out at zero?"""
    # A reference whose sweep never selected a window is null in the stage file
    # and nan once the merge has read it back; either way it is not a k.
    ks = [int(s["argmax_k"]) for s in sweeps.values()
          if s["argmax_k"] is not None and np.isfinite(s["argmax_k"])]
    pos = [k for k in ks if k > 0]
    ratio = (max(pos) / min(pos)) if len(pos) >= 2 else float("nan")
    return {
        "selected_k": ks,
        "max_min_ratio": float(ratio),
        "all_at_grid_floor": bool(ks) and all(k == 0 for k in ks),
        "supported": bool((np.isfinite(ratio) and ratio > 2.0)
                          or (bool(ks) and all(k == 0 for k in ks))),
    }
