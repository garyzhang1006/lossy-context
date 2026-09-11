"""E3: the zero-decay readers, and only then the human counts.

Every reader in this file has retention exactly one at every distance, so any
rejection of ``delta = 0`` is manufactured by the estimator rather than found in
the data.  The two floors say whether the machinery is honest; the three
substantive nulls say whether mismatch that the nuisance parameters cannot
express turns into decay at the human magnitude of mismatch.  The human fit runs
last, after the null outputs are written, because the order is the only thing
keeping the comparison from being adjusted after the fact.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from lcsa.corpusdata import Corpus, PROB_FLOOR
from lcsa.fitting import fit, fit_constrained, local_grid, profile_interval
from lcsa.gates import g4_precision, g5_floors
from lcsa.inference import (cluster_bootstrap, lr_test, rejection_rate, score_test,
                            tost)
from lcsa.kernels import POWER
from lcsa.likelihood import Model, information
from lcsa.projection import (corpus_residual_fraction, explained_share, global_residual,
                             implied_bias, lambda_curvature_leak, split_half_residual)
from lcsa.readers import (calibrate_alpha, calibrate_n0_prime, draw_counts, human_js,
                          null_distributions, reader_n0_prime, tilt_directions)
from lcsa.experiments import Artifacts, jsonable
from lcsa.experiments.e2_ladder import delta_se
from lcsa.experiments.shards import denull, read_shards, write_shard

log = logging.getLogger(__name__)

__all__ = ["h_lexical", "within_passage_rho", "build_h_topic", "build_h_order", "build_nulls",
           "null_replicates", "n0_prime_replicates", "summarise_rates", "replicate_rates",
           "fit_and_profile", "contrast_replicates", "summarise_contrast", "paired_contrast",
           "prepare", "save_prepared", "load_prepared", "run_replicate_shard", "run_human",
           "run_contrast_shard", "assemble", "merge", "run", "FLOORS"]


def h_lexical(corpus: Corpus, model: Model, kernel=POWER, seed: int = 0) -> list[np.ndarray]:
    """The N-LEX tilt: the lexical channel fitted with ``delta`` pinned at zero.

    This is the null the absorption criterion makes a prediction about in both
    directions, since the repaired estimator carries exactly these directions in
    its nuisance span and the naive one does not.
    """
    if not model.lexical:
        raise ValueError("the lexical tilt must be fitted under an estimator that has one")
    f = fit_constrained(corpus, model, kernel, n_starts=3, seed=seed)
    M = corpus.M
    kappa = np.asarray(f.theta[3:3 + M], dtype=np.float64)
    return [t.f @ kappa for t in corpus]


def build_h_topic(corpus: Corpus, words_per_target, contexts, scorer) -> list[np.ndarray]:
    """The N-TOPIC tilt: cosine to a distance-blind mean context embedding.

    The context vector weights every context word equally regardless of distance,
    which is what makes the tilt a pure topic effect with no retention gradient in
    it.  Embeddings come from the reference model's own input table, so the null
    needs no extra checkpoint and no extra GPU pass.
    """
    import torch

    emb = scorer.model.get_input_embeddings().weight.detach().float()

    def vec(word: str) -> np.ndarray:
        ids = scorer.tok.encode(" " + str(word).strip(), add_special_tokens=False)
        if not ids:
            return np.zeros(emb.shape[1], dtype=np.float64)
        return emb[torch.tensor(ids)].mean(dim=0).cpu().numpy().astype(np.float64)

    out = []
    for t, tgt in enumerate(corpus):
        ctx = [w for w in str(contexts[t]).split() if w]
        if ctx:
            v = np.mean([vec(w) for w in ctx], axis=0)
        else:
            v = np.zeros(emb.shape[1], dtype=np.float64)
        nv = np.linalg.norm(v)
        h = np.zeros(tgt.V, dtype=np.float64)
        if nv > 0:
            for i, w in enumerate(words_per_target[t]):
                e = vec(w)
                ne = np.linalg.norm(e)
                h[i] = float(e @ v / (ne * nv)) if ne > 0 else 0.0
        out.append(h)
    return out


def build_h_order(corpus: Corpus, words_per_target, contexts, scorer,
                  phi: float = 0.5, seed: int = 0) -> list[np.ndarray]:
    """The N-ORDER tilt: log-probability change under a partial shuffle.

    A fraction ``phi`` of context positions is permuted and nothing is deleted,
    so the reader still sees every word it ever saw.  That is what makes this the
    sharper of the two substantive nulls: no amount of retention explains it,
    because nothing was forgotten.
    """
    rng = np.random.default_rng(seed)
    out = []
    for t, tgt in enumerate(corpus):
        ctx = [w for w in str(contexts[t]).split() if w]
        if len(ctx) < 2:
            out.append(np.zeros(tgt.V, dtype=np.float64))
            continue
        idx = np.arange(len(ctx))
        k = max(2, int(round(phi * len(ctx))))
        pick = rng.choice(len(ctx), size=min(k, len(ctx)), replace=False)
        perm = rng.permutation(pick)
        idx[pick] = idx[perm]
        shuffled = " ".join(ctx[i] for i in idx)
        words = list(words_per_target[t])
        lp_s = scorer.candidate_logprobs(shuffled, words)
        lp_c = scorer.candidate_logprobs(" ".join(ctx), words)
        out.append(np.asarray(lp_s - lp_c, dtype=np.float64))
    return out


def build_nulls(
    corpus: Corpus,
    theta0: np.ndarray,
    model: Model,
    h_specs: dict[str, list[np.ndarray]],
    kernel=POWER,
    target_js: float | None = None,
    seed: int = 0,
) -> dict:
    """Calibrate each substantive tilt to the human JS and return its ``q_0``.

    The match is like-for-like: the null's divergence is computed from simulated
    counts at the human response counts, so the plug-in bias that makes absolute
    JS unidentified at forty responses cancels on both sides.  The orthogonalised
    unit-scale direction is returned alongside ``q_0`` so that the E4 sweep can
    tilt every row of the cache by the same ``alpha`` without recomputing it.
    """
    js_h = human_js(corpus) if target_js is None else float(target_js)
    out = {"target_js": js_h, "nulls": {}}
    for name, h_list in h_specs.items():
        dirs = tilt_directions(corpus, h_list, theta0, model, orthogonalise=True,
                               kernel=kernel)
        rec = calibrate_alpha(corpus, h_list, js_h, theta0=theta0, model=model,
                              orthogonalise=True, seed=seed, kernel=kernel)
        q = null_distributions(corpus, h_list, rec["alpha"], directions=dirs)
        out["nulls"][name] = {"calibration": rec, "q": q, "directions": dirs}
    return out


def fit_and_profile(corpus: Corpus, model: Model, kernel=POWER, seed: int = 0,
                    profile: bool = True) -> dict:
    """One fit with its cluster-robust test, naive LR and cluster-scaled region."""
    f = fit(corpus, model, kernel, n_starts=3, seed=seed)
    st = score_test(corpus, model, kernel, n_starts=2, seed=seed)
    lr = lr_test(corpus, model, kernel, full_fit=f, null_fit=st.null_fit)
    deff = st.design_effect
    row = {
        "estimator": model.name,
        "delta_hat": float(f.delta),
        "loglik": float(f.loglik),
        "converged": bool(f.success),
        "at_bound": bool(f.at_bound),
        "T_cr1": float(st.T_cr1), "p_cr1": float(st.p_cr1),
        "T_cr2": float(st.T_cr2), "p_cr2": float(st.p_cr2), "df_bm": float(st.df_bm),
        "T_cr3": float(st.T_cr3), "p_cr3": float(st.p_cr3),
        "z": float(st.z),
        "p_one_cr1": float(st.p_one_cr1), "p_one_cr2": float(st.p_one_cr2),
        "p_one_cr3": float(st.p_one_cr3),
        "p_wild": float(st.p_wild), "p_wild_one": float(st.p_wild_one),
        "n_wild": int(st.n_wild), "wild_enumerated": bool(st.wild_enumerated),
        "p_headline": float(st.p),
        "T_raw": float(st.T_raw), "p_raw": float(st.p_raw),
        "design_effect": float(deff),
        "max_leverage": float(np.max(st.leverage)) if st.leverage is not None else float("nan"),
        "LR": float(lr.LR), "p_LR": float(lr.p),
        "n_clusters": int(st.n_clusters),
    }
    if profile:
        scale = 1.0 / deff if np.isfinite(deff) and deff > 0 else 1.0
        g = local_grid(f.delta, se=delta_se(corpus, f.theta, model, kernel, deff), n=21)
        reg = profile_interval(corpus, model, kernel, grid=g, scale=scale, n_starts=1,
                               seed=seed, max_loglik=f.loglik, warm=f.theta)
        row.update({"delta_lo": float(reg.lo), "delta_hi": float(reg.hi),
                    "unbounded_hi": bool(reg.unbounded_hi)})
    return row


# -- replicates ---------------------------------------------------------------

#: Readers whose replicates need only ``theta0`` and the primary estimator.
FLOORS = ("N0", "N0-PRIME")


def _replicate_fit(corp: Corpus, model: Model, kernel, seed: int) -> dict:
    """The score test, the fit and the naive LR on one replicate corpus."""
    try:
        st = score_test(corp, model, kernel, n_starts=1, seed=seed)
        f = fit(corp, model, kernel, n_starts=2, seed=seed)
        lr = lr_test(corp, model, kernel, full_fit=f, null_fit=st.null_fit)
    except Exception as exc:
        return {"failed": True, "error": str(exc)}
    return {
        "failed": False,
        "p_headline": float(st.p), "p_cr1": float(st.p_cr1), "p_cr2": float(st.p_cr2),
        "p_cr3": float(st.p_cr3), "p_one_cr1": float(st.p_one_cr1),
        "p_one_cr2": float(st.p_one_cr2), "p_one_cr3": float(st.p_one_cr3),
        "p_wild": float(st.p_wild), "p_wild_one": float(st.p_wild_one),
        "df_bm": float(st.df_bm),
        "p_LR": float(lr.p), "T_cr1": float(st.T_cr1), "LR": float(lr.LR),
        "delta_hat": float(f.delta), "design_effect": float(st.design_effect),
        "converged": bool(f.success), "at_bound": bool(f.at_bound),
    }


def null_replicates(
    corpus: Corpus,
    q_list: list[np.ndarray],
    models,
    reps=range(200),
    kernel=POWER,
    seed: int = 0,
    progress=None,
) -> list[dict]:
    """One row per (estimator, replicate): counts redrawn from ``q_list``.

    Replicate ``b`` draws its counts from ``seed + b + 1`` under every estimator,
    so the estimators see the same synthetic reader and any range of ``b`` gives
    the same rows whether or not the other replicates were run alongside it.
    """
    rows = []
    reps = list(reps)
    for model in models:
        for i, b in enumerate(reps):
            rng = np.random.default_rng(seed + b + 1)
            corp = corpus.with_counts(draw_counts(corpus, q_list, rng))
            row = _replicate_fit(corp, model, kernel, seed)
            if row["failed"]:
                log.debug("replicate %d failed under %s: %s", b, model.name, row["error"])
            rows.append({"estimator": model.name, "replicate": int(b), **row})
            if progress is not None:
                progress(i + 1, len(reps))
    return rows


def n0_prime_replicates(
    corpus: Corpus,
    theta0: np.ndarray,
    primary: Model,
    sigma: float,
    models,
    reps=range(200),
    kernel=POWER,
    seed: int = 0,
) -> list[dict]:
    """The over-dispersed floor, regenerated per replicate from ``seed + 5000 + b``.

    It carries dependence inside a passage, so its replicates are regenerated
    rather than resampled from one ``q``; the generator is always the primary
    estimator's family at ``theta0``, and every estimator fits the same draw.
    """
    rows = []
    for model in models:
        for b in reps:
            c = reader_n0_prime(corpus, theta0, primary, sigma, seed=seed + 5000 + b,
                                kernel=kernel)
            row = _replicate_fit(c, model, kernel, seed)
            if row["failed"]:
                log.debug("N0-PRIME replicate %d failed under %s: %s", b, model.name,
                          row["error"])
            rows.append({"estimator": model.name, "replicate": int(b), **row})
    return rows


def summarise_rates(rows: list[dict], alpha: float = 0.05) -> list[dict]:
    """Rejection rates per (reader, estimator) from replicate rows, first-seen order.

    A replicate that failed to fit is counted and excluded rather than quietly
    dropped, because a null whose fits fail half the time is a finding about the
    estimator and not a smaller sample.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r.get("reader"), str(r["estimator"])), []).append(r)
    out = []
    for (reader, est), rs in groups.items():
        ok = [r for r in rs if not r.get("failed")]
        p_cr = [r["p_headline"] for r in ok]
        deltas = [r["delta_hat"] for r in ok]
        deffs = [r["design_effect"] for r in ok]
        rate, lo, hi = rejection_rate(p_cr, alpha)

        def _rate(key: str) -> float:
            vals = [r[key] for r in ok if key in r and np.isfinite(r[key])]
            return rejection_rate(vals, alpha)[0] if vals else float("nan")

        rate_lr, lo_lr, hi_lr = rejection_rate([r["p_LR"] for r in ok], alpha)
        dfs = [r["df_bm"] for r in ok if np.isfinite(r.get("df_bm", np.nan))]
        row = {
            "estimator": est,
            "n_replicates": int(len(rs)),
            "n_failed": int(len(rs) - len(ok)),
            "reject_cluster_robust": rate,
            "reject_cr_lo": lo, "reject_cr_hi": hi,
            "reject_wild_two_sided": _rate("p_wild"),
            "reject_cr1": _rate("p_cr1"),
            "reject_cr2_bm": _rate("p_cr2"),
            "reject_cr3": _rate("p_cr3"),
            "reject_one_sided_cr1": _rate("p_one_cr1"),
            "reject_one_sided_cr2_bm": _rate("p_one_cr2"),
            "reject_one_sided_cr3": _rate("p_one_cr3"),
            "median_df_bm": float(np.median(dfs)) if dfs else float("nan"),
            "reject_naive_LR": rate_lr,
            "reject_lr_lo": lo_lr, "reject_lr_hi": hi_lr,
            "median_delta_hat": float(np.median(deltas)) if deltas else float("nan"),
            "iqr_delta_hat": (
                float(np.percentile(deltas, 75) - np.percentile(deltas, 25))
                if len(deltas) > 3 else float("nan")
            ),
            "median_design_effect": float(np.nanmedian(deffs)) if deffs else float("nan"),
        }
        if reader is not None:
            row = {"reader": reader, **row}
        out.append(row)
    return out


def replicate_rates(
    corpus: Corpus,
    q_list: list[np.ndarray],
    models,
    n_rep: int = 200,
    kernel=POWER,
    seed: int = 0,
    alpha: float = 0.05,
    progress=None,
) -> list[dict]:
    """Rejection rates of ``H0: delta = 0`` over ``n_rep`` response-resampling replicates."""
    rows = null_replicates(corpus, q_list, models, range(n_rep), kernel, seed, progress)
    return summarise_rates(rows, alpha)


# -- the paired contrast ------------------------------------------------------


def contrast_replicates(
    human: Corpus,
    null_corpora: dict[str, Corpus],
    model: Model,
    kernel=POWER,
    reps=range(200),
    seed: int = 0,
) -> list[dict]:
    """Human and null ``delta`` refitted on the same passage resample, per replicate."""
    names = list(null_corpora)

    def statistic(sub: Corpus) -> dict:
        idx = np.unique(sub.cluster_index)
        out = {}
        f = fit(sub, model, kernel, n_starts=1, seed=seed)
        out["human"] = f.delta
        for nm in names:
            fn = fit(null_corpora[nm].subset_clusters(idx), model, kernel,
                     n_starts=1, seed=seed)
            out[nm] = fn.delta
        return out

    return cluster_bootstrap(human, statistic, seed=seed, reps=reps)


def summarise_contrast(reps: list[dict], names, margin: float = 0.25) -> dict:
    """TOST on ``log delta_human - log delta_null`` from bootstrap replicate records.

    A replicate where either arm hits the boundary has an undefined log
    difference; it is counted and excluded, never replaced by a number.
    """
    hs = np.array([r.get("human", np.nan) for r in reps], dtype=np.float64)
    arms = [np.array([r.get(nm, np.nan) for r in reps], dtype=np.float64) for nm in names]
    usable = np.isfinite(hs)
    for ns in arms:
        usable &= np.isfinite(ns)
    # Replicates on which every contrast is defined; the per-contrast counts
    # below are the ones each TOST actually used.
    res = {"n_boot": int(len(reps)), "n_usable_replicates": int(usable.sum()),
           "contrasts": {}}
    for nm in names:
        ns = np.array([r.get(nm, np.nan) for r in reps], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.log(hs) - np.log(ns)
        ok = np.isfinite(d)
        # A replicate set that lands on the boundary every time has zero variance,
        # and the pairing correlation is then undefined rather than zero.
        with np.errstate(divide="ignore", invalid="ignore"):
            r_pair = (
                float(np.corrcoef(np.log(hs[ok]), np.log(ns[ok]))[0, 1])
                if ok.sum() > 3 else float("nan")
            )
        t = tost(d[ok], margin=margin)
        res["contrasts"][nm] = {
            "mean_log_diff": float(np.mean(d[ok])) if ok.any() else float("nan"),
            "se_log_diff": float(np.std(d[ok], ddof=1)) if ok.sum() > 1 else float("nan"),
            "pairing_correlation": r_pair,
            "n_usable": int(ok.sum()),
            "n_undefined": int((~ok).sum()),
            "tost_p": t.p,
            "tost_equivalent": bool(t.equivalent),
            "tost_margin": float(margin),
        }
    return res


def paired_contrast(
    human: Corpus,
    null_corpora: dict[str, Corpus],
    model: Model,
    kernel=POWER,
    n_boot: int = 200,
    margin: float = 0.25,
    seed: int = 0,
) -> dict:
    """Paired cluster bootstrap of ``log delta_human - log delta_null``.

    Both arms are refitted on the same resampled passages, which is what makes
    the difference paired and its standard error the one the equivalence test
    needs.
    """
    reps = contrast_replicates(human, null_corpora, model, kernel, range(n_boot), seed)
    return summarise_contrast(reps, list(null_corpora), margin)


# -- stages -------------------------------------------------------------------

PREPARED_JSON = "e3_prepared.json"
PREPARED_NPZ = "e3_prepared.npz"


def prepare(
    corpus: Corpus,
    models,
    out_dir,
    h_specs: dict[str, list[np.ndarray]] | None = None,
    kernel=POWER,
    seed: int = 0,
    readers=None,
) -> dict:
    """Stage one of E3: everything a replicate needs, computed once and saved.

    The constrained fit, the floor's tilt scale and the substantive nulls'
    ``alpha`` are fitted here and never again, so every replicate shard starts
    from the same ``q_0`` and no shard can drift from another.  ``readers``
    restricts the set, which is how the GPT-2-small self-reference run asks for
    the plain floor alone.
    """
    from lcsa.likelihood import evaluate_target

    art = Artifacts(out_dir, "e3")
    primary = models[0]
    want = None if readers is None else set(readers)

    null_fit = fit_constrained(corpus, primary, kernel, n_starts=3, seed=seed)
    theta0 = null_fit.theta.copy()
    theta0[0] = 0.0
    meta: dict[str, dict] = {}
    prepared = {"primary": primary.name, "kernel": kernel.name, "theta0": theta0,
                "n_clusters": int(corpus.n_clusters), "seed": int(seed), "readers": {},
                "calibration": meta}

    if want is None or "N0" in want:
        # The plain floor is i.i.d. from the fitted family at delta = 0, so one q
        # per target is all a replicate needs; the over-dispersed floor cannot be
        # written this way, which is exactly the difference it exists to carry.
        prepared["readers"]["N0"] = {
            "q": [evaluate_target(t, theta0, primary, corpus.M, kernel).q for t in corpus]}
    if want is None or "N0-PRIME" in want:
        st_h = score_test(corpus, primary, kernel, n_starts=2, seed=seed)
        cal = calibrate_n0_prime(corpus, theta0, primary,
                                 target_design_effect=st_h.design_effect,
                                 seed=seed, kernel=kernel)
        meta["N0-PRIME"] = cal
        prepared["readers"]["N0-PRIME"] = {"sigma": float(cal["sigma"])}
    specs = {k: v for k, v in (h_specs or {}).items() if want is None or k in want}
    if specs:
        built = build_nulls(corpus, theta0, primary, specs, kernel=kernel, seed=seed)
        meta["substantive"] = {k: v["calibration"] for k, v in built["nulls"].items()}
        meta["target_js"] = built["target_js"]
        for nm, rec in built["nulls"].items():
            prepared["readers"][nm] = {"q": rec["q"], "directions": rec["directions"],
                                       "alpha": float(rec["calibration"]["alpha"])}
    save_prepared(out_dir, prepared)
    art.save("e3_calibration", meta)
    return prepared


def save_prepared(out_dir, prepared: dict) -> None:
    """Numbers to a flat ``.npz``, names and calibration records to JSON."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"theta0": np.asarray(prepared["theta0"], dtype=np.float64)}
    meta = {"primary": prepared["primary"], "kernel": prepared.get("kernel", "power"),
            "n_clusters": prepared["n_clusters"], "seed": prepared["seed"],
            "calibration": prepared["calibration"], "readers": {}}
    for nm, rec in prepared["readers"].items():
        m = {k: v for k, v in rec.items() if k not in ("q", "directions")}
        for key in ("q", "directions"):
            if key in rec:
                arrays[f"{key}__{nm}"] = np.concatenate(
                    [np.asarray(x, dtype=np.float64) for x in rec[key]])
                m[key] = True
        meta["readers"][nm] = m
    np.savez_compressed(out / PREPARED_NPZ, **arrays)
    (out / PREPARED_JSON).write_text(json.dumps(jsonable(meta), indent=2))


def load_prepared(out_dir, corpus: Corpus) -> dict:
    """Inverse of :func:`save_prepared`; splits the flat arrays by the corpus's ``V``."""
    from pathlib import Path

    out = Path(out_dir)
    if not (out / PREPARED_JSON).exists():
        raise FileNotFoundError(
            f"{out / PREPARED_JSON} is missing; run `lcsa e3 --stage prepare` first")
    meta = json.loads((out / PREPARED_JSON).read_text())
    sizes = [t.V for t in corpus]
    cuts = np.cumsum(sizes)[:-1]
    total = int(sum(sizes))
    with np.load(out / PREPARED_NPZ, allow_pickle=False) as z:
        prepared = {"primary": meta["primary"], "kernel": meta.get("kernel", "power"),
                    "theta0": z["theta0"].astype(np.float64),
                    "n_clusters": int(meta["n_clusters"]), "seed": int(meta["seed"]),
                    "calibration": denull(meta["calibration"]), "readers": {}}
        for nm, m in meta["readers"].items():
            rec = {k: v for k, v in m.items() if k not in ("q", "directions")}
            for key in ("q", "directions"):
                if m.get(key):
                    flat = z[f"{key}__{nm}"]
                    if flat.size != total:
                        raise ValueError(
                            f"{out / PREPARED_NPZ} holds {flat.size} entries for {nm} but "
                            f"the cache has {total} candidates; the prepared file and the "
                            "cache come from different builds")
                    rec[key] = [np.ascontiguousarray(a) for a in np.split(flat, cuts)]
            prepared["readers"][nm] = rec
    return prepared


def _check_primary(prepared: dict, models, kernel=None) -> Model:
    primary = models[0]
    if primary.name != prepared["primary"]:
        raise ValueError(
            f"the prepared stage used {prepared['primary']!r} as the primary estimator "
            f"but this run lists {primary.name!r} first; pass the same --estimators")
    want = prepared.get("kernel", "power")
    if kernel is not None and kernel.name != want:
        raise ValueError(
            f"the prepared stage in this directory used the {want!r} kernel but this run "
            f"asks for {kernel.name!r}; theta0 and the floors depend on the kernel, so "
            f"pass --kernel {want} or prepare into a different --out")
    return primary


def run_replicate_shard(
    corpus: Corpus,
    prepared: dict,
    models,
    out_dir,
    reader: str,
    reps: range,
    kernel=POWER,
    seed: int = 0,
) -> list[dict]:
    """Stage two of E3: replicates ``reps`` of one reader under every estimator."""
    primary = _check_primary(prepared, models, kernel)
    if reader not in prepared["readers"]:
        raise KeyError(f"reader {reader!r} was not prepared; have {list(prepared['readers'])}")
    rec = prepared["readers"][reader]
    if reader == "N0-PRIME":
        rows = n0_prime_replicates(corpus, prepared["theta0"], primary, rec["sigma"], models,
                                   reps, kernel=kernel, seed=seed)
    else:
        rows = null_replicates(corpus, rec["q"], models, reps, kernel=kernel, seed=seed)
    rows = [{"reader": reader, **r} for r in rows]
    write_shard(out_dir, f"e3_rates_{reader}", reps, rows)
    return rows


def null_corpora_from(corpus: Corpus, prepared: dict, seed: int) -> dict[str, Corpus]:
    """One drawn corpus per substantive null, the arm the paired contrast refits."""
    out = {}
    for nm, rec in prepared["readers"].items():
        if nm in FLOORS:
            continue
        rng = np.random.default_rng(seed + 99)
        out[nm] = corpus.with_counts(draw_counts(corpus, rec["q"], rng))
    return out


#: Jeffreys smoothing values at which every absorption fraction is reported.
ALPHAS = (0.1, 0.5, 1.0)
#: Random split-half draws averaged in the debiased fractions.
N_SPLITS = 20
#: Cluster bootstrap draws behind the implied-bias interval.
N_BIAS_BOOT = 200
#: Floor replicates used to check that the debiasing removes pure noise.
N_RULE_FLOOR = 10
#: The registered thresholds of the reading rule.
RULE = {"floor_signal_cap": 0.10, "outside_at_least": 0.50, "absorbable_at_most": 0.25}


def reading_rule(corpus: Corpus, prepared: dict, primary: Model, debiased: list[dict],
                 kernel=POWER, seed: int = 0) -> dict:
    """The registered reading of the human residual, decided before unblinding.

    The verb the abstract is allowed to use depends on one number, the
    split-half debiased global residual fraction of the human mismatch under
    the naive estimator at ``alpha = 0.5``, and on one sanity check, that the
    same debiasing leaves at most a tenth of the squared norm as signal on the
    plain floor, where the mismatch is sampling noise by construction.  A
    fraction of at least one half reads as "outside the tangent space", at
    most a quarter as "absorbable", and anything between gets no headline verb.
    """
    q0 = prepared["readers"].get("N0", {}).get("q")
    floor_shares = []
    if q0 is not None:
        for b in range(N_RULE_FLOOR):
            rng = np.random.default_rng([int(seed), 4242, b])
            corp = corpus.with_counts(draw_counts(corpus, q0, rng))
            nm = fit_constrained(corp, primary, kernel, n_starts=1, seed=seed)
            sh = split_half_residual(corp, nm.theta, primary, kernel, alpha=0.5,
                                     n_splits=N_SPLITS, seed=seed + b)
            floor_shares.append(sh["signal_share_of_norm2"])
    floor_share = float(np.nanmedian(floor_shares)) if floor_shares else float("nan")
    human = next((r["fraction_debiased"] for r in debiased
                  if r["estimator"] == primary.name and r["jeffreys_alpha"] == 0.5),
                 float("nan"))
    floor_ok = bool(np.isfinite(floor_share) and floor_share <= RULE["floor_signal_cap"])
    if not floor_ok or not np.isfinite(human):
        verdict = "void"
    elif human >= RULE["outside_at_least"]:
        verdict = "outside"
    elif human <= RULE["absorbable_at_most"]:
        verdict = "absorbable"
    else:
        verdict = "indeterminate"
    return {"verdict": verdict, "human_fraction_debiased": float(human),
            "floor_signal_share_median": floor_share, "floor_signal_shares": floor_shares,
            "floor_check_passed": floor_ok, "thresholds": dict(RULE),
            "estimator": primary.name, "jeffreys_alpha": 0.5}


def run_human(
    corpus: Corpus,
    prepared: dict,
    models,
    out_dir,
    kernel=POWER,
    seed: int = 0,
) -> dict:
    """Stage three of E3: the human fits, residual fractions and precision inputs.

    It runs after the null outputs are written, because the order is the only
    thing keeping the comparison from being adjusted after the fact.
    """
    _check_primary(prepared, models, kernel)
    art = Artifacts(out_dir, "e3")
    fits = [fit_and_profile(corpus, m, kernel, seed=seed) for m in models]
    art.table("e3_human_fit", fits)
    # Every absorption statistic is evaluated at each estimator's own
    # constrained null, since the tangent space it projects against is that
    # estimator's.  The global fraction is the quantity in Proposition 2; the
    # per-target fraction is kept as the lower bound it is.
    resid, debiased, bias_rows = [], [], []
    for m in models:
        nm = fit_constrained(corpus, m, kernel, n_starts=2, seed=seed)
        g = global_residual(corpus, nm.theta, m, kernel)
        rep = corpus_residual_fraction(corpus, nm.theta, m, kernel)
        resid.append({
            "estimator": m.name,
            "residual_fraction_global": g.fraction,
            "residual_fraction_per_target": g.fraction_per_target,
            "residual_fraction_unweighted": rep.fraction_unweighted,
            "alignment": g.alignment,
            "span_dim_mean": rep.span_dim_mean,
            "n_targets": g.n_targets_used,
        })
        for a in ALPHAS:
            sh = split_half_residual(corpus, nm.theta, m, kernel, alpha=a,
                                     n_splits=N_SPLITS, seed=seed)
            debiased.append({"estimator": m.name, "jeffreys_alpha": float(a), **sh})
        ib = implied_bias(corpus, nm.theta, m, kernel, n_boot=N_BIAS_BOOT, seed=seed)
        lk = lambda_curvature_leak(corpus, nm.theta, m, kernel)
        row = {"estimator": m.name, **ib,
               "lambda_coefficient": lk["a_lambda"],
               "delta_leak_second_order": lk["delta_leak_second_order"],
               "leak_over_first_order": lk["leak_over_first_order"]}
        dirs = {k: prepared["readers"][k]["directions"] for k in ("N-TOPIC", "N-ORDER")
                if k in prepared["readers"] and "directions" in prepared["readers"][k]}
        if dirs:
            es = explained_share(corpus, nm.theta, m, dirs, kernel)
            row.update({"share_explained_topic_order": es["share_explained"],
                        "share_directions": es["directions"],
                        "share_coefficients": es["coefficients"],
                        "share_gram_condition": es["gram_condition"]})
        bias_rows.append(row)
    art.table("e3_human_residual", resid)
    art.table("e3_human_residual_debiased", debiased)
    art.table("e3_human_implied_bias", bias_rows)
    rule = reading_rule(corpus, prepared, models[0], debiased, kernel, seed)
    art.save("e3_reading_rule", rule)
    precision = {
        "sd_passage_log_delta": passage_sd_log_delta(corpus, models[0], kernel, seed),
        "within_passage_rho": within_passage_rho(corpus, prepared["theta0"], models[0], kernel),
    }
    human = {"fits": fits, "residual_fraction": resid, "residual_debiased": debiased,
             "implied_bias": bias_rows, "reading_rule": rule, "precision": precision}
    art.save("e3_human_stage", human)
    return human


def run_contrast_shard(
    corpus: Corpus,
    prepared: dict,
    models,
    out_dir,
    reps: range,
    kernel=POWER,
    seed: int = 0,
) -> list[dict]:
    """Stage four of E3: paired bootstrap replicates ``reps`` of the contrast."""
    primary = _check_primary(prepared, models, kernel)
    nulls = null_corpora_from(corpus, prepared, seed)
    recs = contrast_replicates(corpus, nulls, primary, kernel, reps, seed) if nulls else []
    write_shard(out_dir, "e3_contrast", reps, recs)
    return recs


def assemble(
    prepared: dict,
    rate_rows: list[dict],
    out_dir,
    human: dict | None = None,
    contrast_reps: list[dict] | None = None,
    margin: float = 0.25,
    alpha: float = 0.05,
) -> dict:
    """Stage five of E3: the registered tables, gate G5, then G4 from the contrast."""
    art = Artifacts(out_dir, "e3")
    rows = summarise_rates(rate_rows, alpha)
    art.table("e3_rejection_rates", rows)
    floors = {
        f"{r['reader']}/{r['estimator']}": r["reject_cluster_robust"]
        for r in rows if r["reader"] in FLOORS
    }
    res = {"rejection_rates": rows, "calibration": prepared["calibration"],
           "g5": g5_floors(floors), "human": None}
    art.save("e3_nulls", res)
    if human is not None:
        names = [nm for nm in prepared["readers"] if nm not in FLOORS]
        contrast = (summarise_contrast(contrast_reps, names, margin)
                    if contrast_reps and names else None)
        r_pair = float("nan")
        if contrast and contrast.get("contrasts"):
            vals = [v["pairing_correlation"] for v in contrast["contrasts"].values()]
            vals = [v for v in vals if np.isfinite(v)]
            r_pair = float(np.mean(vals)) if vals else float("nan")
        prec = human["precision"]
        res["human"] = {
            "fits": human["fits"],
            "residual_fraction": human["residual_fraction"],
            "residual_debiased": human.get("residual_debiased"),
            "implied_bias": human.get("implied_bias"),
            "reading_rule": human.get("reading_rule"),
            "contrast": contrast,
            "g4": g4_precision(prec["sd_passage_log_delta"], r_pair,
                               prec["within_passage_rho"],
                               n_clusters=int(prepared["n_clusters"])),
        }
        art.save("e3_human", res["human"])
    art.save("e3_summary", res)
    return res


def merge(out_dir, margin: float = 0.25, alpha: float = 0.05, n_rep: int | None = None,
          n_boot: int | None = None, require_human: bool = False) -> dict:
    """Combine the prepared stage, the rate shards and the human stage on disk.

    ``require_human`` makes an absent human stage or contrast an error; the
    registered merge passes it so that a leg the paper quotes cannot be missing
    without the merge saying so.
    """
    from pathlib import Path

    out = Path(out_dir)
    if not (out / PREPARED_JSON).exists():
        raise FileNotFoundError(
            f"{out / PREPARED_JSON} is missing; run `lcsa e3 --stage prepare` first")
    meta = json.loads((out / PREPARED_JSON).read_text())
    prepared = {"primary": meta["primary"], "n_clusters": int(meta["n_clusters"]),
                "calibration": denull(meta["calibration"]), "readers": meta["readers"]}
    rate_rows = []
    for nm in meta["readers"]:
        rate_rows += read_shards(out, f"e3_rates_{nm}", n_rep)
    human = contrast = None
    if (out / "e3_human_stage.json").exists():
        human = denull(json.loads((out / "e3_human_stage.json").read_text()))
        try:
            contrast = read_shards(out, "e3_contrast", n_boot)
        except FileNotFoundError:
            if require_human:
                raise
            contrast = None
    elif require_human:
        raise FileNotFoundError(
            f"{out / 'e3_human_stage.json'} is missing; the human fit is registered, run "
            "`lcsa e3 --stage human` (slurm/e3_human.sbatch task 0) before merging")
    return assemble(prepared, rate_rows, out, human, contrast, margin, alpha)


def run(
    corpus: Corpus,
    models,
    out_dir,
    h_specs: dict[str, list[np.ndarray]] | None = None,
    kernel=POWER,
    n_rep: int = 200,
    n_boot: int = 200,
    seed: int = 0,
    fit_human: bool = True,
    readers=None,
) -> dict:
    """Full E3 leg in one process: floors, substantive nulls, G5, then the human fit."""
    prepared = prepare(corpus, models, out_dir, h_specs, kernel=kernel, seed=seed,
                       readers=readers)
    rows = []
    for nm in prepared["readers"]:
        rows += run_replicate_shard(corpus, prepared, models, out_dir, nm, range(n_rep),
                                    kernel=kernel, seed=seed)
    human = contrast = None
    if fit_human:
        human = run_human(corpus, prepared, models, out_dir, kernel=kernel, seed=seed)
        contrast = run_contrast_shard(corpus, prepared, models, out_dir, range(n_boot),
                                      kernel=kernel, seed=seed)
    return assemble(prepared, rows, out_dir, human, contrast)


def within_passage_rho(corpus: Corpus, theta0: np.ndarray, model: Model,
                       kernel=POWER) -> float:
    """Intraclass correlation of the per-target efficient scores within a passage.

    This is the quantity gate G4 caps, and it is what decides whether 55 passages
    behave like 55 independent units or like fewer.  It is computed from the
    efficient score rather than from the raw one, because the raw score carries
    nuisance variation that the test projects out before it ever sees it.
    """
    from lcsa.likelihood import observed_scores

    U = observed_scores(corpus, theta0, model, kernel)
    I = information(corpus, theta0, model, kernel)
    try:
        b = np.linalg.pinv(I[1:, 1:]) @ I[1:, 0]
    except np.linalg.LinAlgError:
        return float("nan")
    u = U[:, 0] - U[:, 1:] @ b
    g = corpus.cluster_index
    groups = [u[g == c] for c in np.unique(g)]
    groups = [x for x in groups if x.size > 1]
    k = len(groups)
    if k < 2:
        return float("nan")
    n_tot = sum(x.size for x in groups)
    m_bar = n_tot / k
    grand = float(np.mean(np.concatenate(groups)))
    ssb = sum(x.size * (x.mean() - grand) ** 2 for x in groups)
    ssw = sum(float(((x - x.mean()) ** 2).sum()) for x in groups)
    msb = ssb / (k - 1)
    msw = ssw / max(n_tot - k, 1)
    den = msb + (m_bar - 1) * msw
    return float((msb - msw) / den) if den > 0 else float("nan")


def passage_sd_log_delta(corpus: Corpus, model: Model, kernel, seed: int) -> float:
    """Passage-level SD of ``log delta`` from one fit per passage, the input to G4."""
    logs = []
    for c in np.unique(corpus.cluster_index):
        try:
            f = fit(corpus.subset_clusters([c]), model, kernel, n_starts=1, seed=seed)
        except Exception:
            continue
        if f.delta > 0 and np.isfinite(f.delta):
            logs.append(np.log(f.delta))
    return float(np.std(logs, ddof=1)) if len(logs) > 2 else float("nan")
