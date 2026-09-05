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

import logging

import numpy as np

from lcsa.corpusdata import Corpus, PROB_FLOOR
from lcsa.fitting import fit, fit_constrained, local_grid, profile_interval
from lcsa.gates import g4_precision, g5_floors
from lcsa.inference import (cluster_bootstrap, lr_test, rejection_rate, score_test,
                            tost)
from lcsa.kernels import POWER
from lcsa.likelihood import Model, information
from lcsa.projection import corpus_residual_fraction
from lcsa.readers import (calibrate_alpha, calibrate_n0_prime, draw_counts, human_js,
                          null_distributions, reader_n0_prime)
from lcsa.experiments import Artifacts
from lcsa.experiments.e2_ladder import delta_se

log = logging.getLogger(__name__)

__all__ = ["h_lexical", "within_passage_rho", "build_h_topic", "build_h_order", "build_nulls",
           "replicate_rates", "fit_and_profile", "paired_contrast", "run"]


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
    JS unidentified at forty responses cancels on both sides.
    """
    js_h = human_js(corpus) if target_js is None else float(target_js)
    out = {"target_js": js_h, "nulls": {}}
    for name, h_list in h_specs.items():
        rec = calibrate_alpha(corpus, h_list, js_h, theta0=theta0, model=model,
                              orthogonalise=True, seed=seed, kernel=kernel)
        q = null_distributions(corpus, h_list, rec["alpha"], theta0=theta0, model=model,
                               orthogonalise=True, kernel=kernel)
        out["nulls"][name] = {"calibration": rec, "q": q}
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
        "T_cr3": float(st.T_cr3), "p_cr3": float(st.p_cr3),
        "p_headline": float(st.p),
        "T_raw": float(st.T_raw), "p_raw": float(st.p_raw),
        "design_effect": float(deff),
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
    """Rejection rates of ``H0: delta = 0`` over response-resampling replicates.

    A replicate that fails to fit is counted and excluded rather than quietly
    dropped, because a null whose fits fail half the time is a finding about the
    estimator and not a smaller sample.
    """
    rows = []
    for model in models:
        p_cr, p_lr, p_cr3, deltas, deffs, fails = [], [], [], [], [], 0
        for b in range(n_rep):
            rng = np.random.default_rng(seed + b + 1)
            corp = corpus.with_counts(draw_counts(corpus, q_list, rng))
            try:
                st = score_test(corp, model, kernel, n_starts=1, seed=seed)
                f = fit(corp, model, kernel, n_starts=2, seed=seed)
                lr = lr_test(corp, model, kernel, full_fit=f, null_fit=st.null_fit)
            except Exception as exc:
                log.debug("replicate %d failed under %s: %s", b, model.name, exc)
                fails += 1
                continue
            p_cr.append(st.p)
            p_cr3.append(st.p_cr3)
            p_lr.append(lr.p)
            deltas.append(f.delta)
            deffs.append(st.design_effect)
            if progress is not None:
                progress(b + 1, n_rep)
        rate, lo, hi = rejection_rate(p_cr, alpha)
        rate_lr, lo_lr, hi_lr = rejection_rate(p_lr, alpha)
        rate3, _, _ = rejection_rate(p_cr3, alpha)
        rows.append({
            "estimator": model.name,
            "n_replicates": int(n_rep),
            "n_failed": int(fails),
            "reject_cluster_robust": rate,
            "reject_cr_lo": lo, "reject_cr_hi": hi,
            "reject_cr3": rate3,
            "reject_naive_LR": rate_lr,
            "reject_lr_lo": lo_lr, "reject_lr_hi": hi_lr,
            "median_delta_hat": float(np.median(deltas)) if deltas else float("nan"),
            "iqr_delta_hat": (
                float(np.percentile(deltas, 75) - np.percentile(deltas, 25))
                if len(deltas) > 3 else float("nan")
            ),
            "median_design_effect": float(np.nanmedian(deffs)) if deffs else float("nan"),
        })
    return rows


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
    needs.  A replicate where either arm hits the boundary has an undefined log
    difference; it is counted and excluded, never replaced by a number.
    """
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

    reps = cluster_bootstrap(human, statistic, n_boot=n_boot, seed=seed)
    res = {"n_boot": int(n_boot), "n_usable_replicates": len(reps), "contrasts": {}}
    hs = np.array([r.get("human", np.nan) for r in reps], dtype=np.float64)
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
) -> dict:
    """Full E3 leg: floors, substantive nulls, gate G5, then the human fit."""
    art = Artifacts(out_dir, "e3")
    primary = models[0]
    null_fit = fit_constrained(corpus, primary, kernel, n_starts=3, seed=seed)
    theta0 = null_fit.theta.copy()
    theta0[0] = 0.0

    meta: dict[str, dict] = {}

    from lcsa.likelihood import evaluate_target

    # The plain floor is i.i.d. from the fitted family at delta = 0, so one q per
    # target is all a replicate needs; the over-dispersed floor below cannot be
    # written this way, which is exactly the difference it exists to carry.
    q0 = [evaluate_target(t, theta0, primary, corpus.M, kernel).q for t in corpus]

    st_h = score_test(corpus, primary, kernel, n_starts=2, seed=seed)
    cal = calibrate_n0_prime(corpus, theta0, primary, target_design_effect=st_h.design_effect,
                             seed=seed, kernel=kernel)
    meta["N0-PRIME"] = cal
    rows = []
    rows += [dict(r, reader="N0") for r in
             replicate_rates(corpus, q0, models, n_rep=n_rep, kernel=kernel, seed=seed)]
    # The over-dispersed floor carries dependence inside a passage, so its
    # replicates are regenerated rather than resampled from one q.
    p_rows = []
    for model in models:
        p_cr, p_lr, deltas = [], [], []
        for b in range(n_rep):
            c = reader_n0_prime(corpus, theta0, primary, cal["sigma"],
                                seed=seed + 5000 + b, kernel=kernel)
            try:
                st = score_test(c, model, kernel, n_starts=1, seed=seed)
                f = fit(c, model, kernel, n_starts=2, seed=seed)
                lr = lr_test(c, model, kernel, full_fit=f, null_fit=st.null_fit)
            except Exception as exc:
                log.debug("N0-PRIME replicate %d failed: %s", b, exc)
                continue
            p_cr.append(st.p)
            p_lr.append(lr.p)
            deltas.append(f.delta)
        rate, lo, hi = rejection_rate(p_cr)
        rate_lr, lo_lr, hi_lr = rejection_rate(p_lr)
        p_rows.append({
            "reader": "N0-PRIME", "estimator": model.name,
            "n_replicates": int(n_rep), "n_failed": int(n_rep - len(p_cr)),
            "reject_cluster_robust": rate, "reject_cr_lo": lo, "reject_cr_hi": hi,
            "reject_naive_LR": rate_lr, "reject_lr_lo": lo_lr, "reject_lr_hi": hi_lr,
            "median_delta_hat": float(np.median(deltas)) if deltas else float("nan"),
        })
    rows += p_rows

    null_corpora: dict[str, Corpus] = {}
    if h_specs:
        built = build_nulls(corpus, theta0, primary, h_specs, kernel=kernel, seed=seed)
        meta["substantive"] = {k: v["calibration"] for k, v in built["nulls"].items()}
        meta["target_js"] = built["target_js"]
        for nm, rec in built["nulls"].items():
            rows += [dict(r, reader=nm) for r in
                     replicate_rates(corpus, rec["q"], models, n_rep=n_rep,
                                     kernel=kernel, seed=seed)]
            rng = np.random.default_rng(seed + 99)
            null_corpora[nm] = corpus.with_counts(draw_counts(corpus, rec["q"], rng))

    art.table("e3_rejection_rates", rows)

    floors = {
        f"{r['reader']}/{r['estimator']}": r["reject_cluster_robust"]
        for r in rows if r["reader"] in ("N0", "N0-PRIME")
    }
    res = {"rejection_rates": rows, "calibration": meta,
           "g5": g5_floors(floors), "human": None}
    art.save("e3_nulls", res)

    if fit_human:
        human_rows = [fit_and_profile(corpus, m, kernel, seed=seed) for m in models]
        art.table("e3_human_fit", human_rows)
        # The residual fraction is evaluated at each estimator's own constrained
        # null, since the span it projects against is that estimator's span.
        resid = []
        for m in models:
            nm = fit_constrained(corpus, m, kernel, n_starts=2, seed=seed)
            rep = corpus_residual_fraction(corpus, nm.theta, m, kernel)
            resid.append({
                "estimator": m.name,
                "residual_fraction": rep.fraction,
                "residual_fraction_unweighted": rep.fraction_unweighted,
                "span_dim_mean": rep.span_dim_mean,
                "n_targets": rep.n_targets_used,
            })
        art.table("e3_human_residual", resid)
        contrast = (
            paired_contrast(corpus, null_corpora, primary, kernel, n_boot=n_boot, seed=seed)
            if null_corpora else None
        )
        sd_pass, r_pair, rho = _precision_inputs(corpus, primary, kernel, contrast, seed,
                                                 theta0=theta0)
        res["human"] = {
            "fits": human_rows,
            "residual_fraction": resid,
            "contrast": contrast,
            "g4": g4_precision(sd_pass, r_pair, rho, n_clusters=corpus.n_clusters),
        }
        art.save("e3_human", res["human"])
    art.save("e3_summary", res)
    return res


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


def _precision_inputs(corpus: Corpus, model: Model, kernel, contrast, seed: int,
                      theta0: np.ndarray | None = None):
    """Passage-level SD of ``log delta``, the pairing correlation and within-passage rho."""
    logs = []
    for c in np.unique(corpus.cluster_index):
        try:
            f = fit(corpus.subset_clusters([c]), model, kernel, n_starts=1, seed=seed)
        except Exception:
            continue
        if f.delta > 0 and np.isfinite(f.delta):
            logs.append(np.log(f.delta))
    sd = float(np.std(logs, ddof=1)) if len(logs) > 2 else float("nan")
    r_pair = float("nan")
    if contrast and contrast.get("contrasts"):
        vals = [v["pairing_correlation"] for v in contrast["contrasts"].values()]
        vals = [v for v in vals if np.isfinite(v)]
        r_pair = float(np.mean(vals)) if vals else float("nan")
    if theta0 is None:
        theta0 = fit_constrained(corpus, model, kernel, n_starts=2, seed=seed).theta
    rho = within_passage_rho(corpus, theta0, model, kernel)
    return sd, r_pair, rho
