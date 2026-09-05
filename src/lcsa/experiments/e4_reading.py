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

import logging

import numpy as np

from lcsa.baselines import context_slopes, hard_window_sweep, sweep_disagreement
from lcsa.corpusdata import Corpus
from lcsa.kernels import POWER
from lcsa.likelihood import Model
from lcsa.readingtime import RTResult, _fit_ll, _gauss_ll, reading_time_gain, spillover
from lcsa.experiments import Artifacts

log = logging.getLogger(__name__)

__all__ = ["K_GRID", "gaze_table", "window_surprisal", "heldout_delta_ll", "sweep_reference",
           "argmax_bootstrap", "rt_gain_table", "run"]

#: The context-limitation grid, in words of preceding context.
K_GRID = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32)


def gaze_table(provo, keys, unigrams=None, measure: str = "gaze"):
    """Align the eye-tracking arm to the built targets.

    Returns the mean gaze duration per target, the control matrix (log unigram
    frequency and word length) and the passage label, in the corpus's own target
    order.  A target with no eye-tracking record becomes ``nan`` and is dropped
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
    y, ctrl, passage = [], [], []
    for tid, wn in keys:
        y.append(lut.get((int(tid), int(wn)), np.nan))
        w = wmap.get((int(tid), int(wn)), "")
        lf = float(unigrams(w)) if unigrams is not None else 0.0
        ctrl.append([lf, float(len(w))])
        passage.append(int(tid))
    return (np.asarray(y, dtype=np.float64),
            np.asarray(ctrl, dtype=np.float64),
            np.asarray(passage))


def window_surprisal(corpus: Corpus, k: int, P_override=None) -> np.ndarray:
    """``-log p_ref(w_t | last k words)`` for each target's own word.

    ``P_override`` supplies an alternative cache, which is how the tilted
    zero-decay references enter the sweep at no marginal GPU cost: they are the
    same ablation block with an exponential tilt applied to every row.
    """
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
) -> float:
    """Held-out log-likelihood gain from adding one predictor and its spillover.

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
    full = np.column_stack([base, x, spillover(x, passage)])
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
) -> dict:
    """The whole delta-log-likelihood curve for one reference, plus its argmax."""
    curve = []
    for k in k_grid:
        s = window_surprisal(corpus, int(k), P_override)
        curve.append({"k": int(k),
                      "delta_ll": heldout_delta_ll(gaze, controls, s, passage,
                                                   n_folds, use_mixed, seed)})
    fin = [c for c in curve if np.isfinite(c["delta_ll"])]
    best = max(fin, key=lambda c: c["delta_ll"])["k"] if fin else None
    return {"reference": name, "curve": curve, "argmax_k": best,
            "k_grid": list(int(k) for k in k_grid)}


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
) -> dict:
    """Cluster bootstrap of the selected window.

    An argmax over a discrete grid is lumpy under resampling, so the honest
    summary is the distribution over grid points rather than a point estimate
    with an interval drawn around it.
    """
    passage = np.asarray(passage)
    uniq = np.unique(passage)
    surp = {int(k): window_surprisal(corpus, int(k), P_override) for k in k_grid}
    picks = []
    for b in range(n_boot):
        rng = np.random.default_rng(seed + b + 1)
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        rows = np.concatenate([np.flatnonzero(passage == p) for p in pick])
        best_k, best_v = None, -np.inf
        for k in k_grid:
            v = heldout_delta_ll(gaze[rows], np.asarray(controls)[rows],
                                 surp[int(k)][rows], passage[rows], n_folds,
                                 use_mixed, seed)
            if np.isfinite(v) and v > best_v:
                best_k, best_v = int(k), v
        if best_k is not None:
            picks.append(best_k)
    if not picks:
        return {"n_boot": int(n_boot), "n_usable": 0, "distribution": {}}
    vals, counts = np.unique(np.asarray(picks), return_counts=True)
    return {
        "n_boot": int(n_boot),
        "n_usable": len(picks),
        "distribution": {int(v): int(c) for v, c in zip(vals, counts)},
        "mode_k": int(vals[int(np.argmax(counts))]),
        "median_k": float(np.median(picks)),
    }


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
                                        n_folds=n_folds, use_mixed=use_mixed, seed=seed)
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
        })
    human = next((r["gain_per_word"] for r in rows if r["reader"] == "human"), None)
    for r in rows:
        r["fraction_of_human_gain"] = (
            float(r["gain_per_word"] / human)
            if human not in (None, 0.0) and np.isfinite(r["gain_per_word"]) else float("nan")
        )
    return rows


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
    seed: int = 0,
) -> dict:
    """Full E4 leg: the sweep across references, the argmax bootstrap, the gains."""
    art = Artifacts(out_dir, "e4")
    refs = {"primary": None} if not references else references
    sweeps, flat = {}, []
    for name, override in refs.items():
        s = sweep_reference(corpus, gaze, controls, passage, name, override, k_grid,
                            n_folds, use_mixed, seed)
        sweeps[name] = s
        for c in s["curve"]:
            flat.append({"reference": name, **c})
    art.table("e4_sweep_curves", flat)

    boots = {
        name: argmax_bootstrap(corpus, gaze, controls, passage, refs[name], k_grid,
                               n_boot=n_boot, seed=seed)
        for name in refs
    }
    res = {
        "sweeps": sweeps,
        "argmax_bootstrap": boots,
        "selected": {n: s["argmax_k"] for n, s in sweeps.items()},
        "prediction_8": _prediction_8(sweeps),
        "model_free": context_slopes(corpus),
    }
    if fitted:
        rows = rt_gain_table(corpus, gaze, controls, passage, fitted, n_folds=5,
                             use_mixed=use_mixed, seed=seed)
        art.table("e4_rt_gain", rows)
        res["rt_gain"] = rows
    hw = hard_window_sweep(corpus, models[0], windows=tuple(k_grid), n_folds=5, seed=seed)
    art.table("e4_hard_window_likelihood", hw)
    res["hard_window_likelihood"] = [h.__dict__ for h in hw]
    res["hard_window_agreement"] = sweep_disagreement({"cloze": hw})
    art.save("e4_summary", res)
    return res


def _prediction_8(sweeps: dict) -> dict:
    """Does the sweep disagree across zero-decay references, or bottom out at zero?"""
    ks = [s["argmax_k"] for s in sweeps.values() if s["argmax_k"] is not None]
    pos = [k for k in ks if k > 0]
    ratio = (max(pos) / min(pos)) if len(pos) >= 2 else float("nan")
    return {
        "selected_k": ks,
        "max_min_ratio": float(ratio),
        "all_at_grid_floor": bool(ks) and all(k == 0 for k in ks),
        "supported": bool((np.isfinite(ratio) and ratio > 2.0)
                          or (bool(ks) and all(k == 0 for k in ks))),
    }
