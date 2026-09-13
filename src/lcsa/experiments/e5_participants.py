"""E5, the participant audit: does the fitted half-life vary between readers?

A pooled half-life is one number for 470 people, and a claim that it is a
property of readers rather than of the reference model needs the estimate to
move between people in a way that survives a split of each person's own
responses.  This leg fits ``delta`` per participant with the nuisances pinned
at the pooled constrained values, so that between-participant spread cannot
come from nuisance drift, and again with the nuisances free; it then reports
the split-half reliability of the participant estimates, the between-
participant spread under each estimator, and the rank correlation of each
participant's estimate with the alignment of that participant's mismatch to
the topic and order directions of E3.

The primary route to the same question is the participant-level fits of
Li and colleagues (2026); when those are supplied the two sets of estimates
are compared on the participants they share.  This module is the fallback and
runs on the raw cloze export, which is not part of the distributed norms.
"""

from __future__ import annotations

import logging
import zlib

import numpy as np

from lcsa.corpusdata import Corpus
from lcsa.data.provo import canonical_word
from lcsa.experiments import Artifacts
from lcsa.experiments.shards import denull, read_shards, write_shard
from lcsa.fitting import fit, fit_constrained
from lcsa.kernels import POWER, d_half_from_delta
from lcsa.likelihood import Model, evaluate_target
from lcsa.projection import centre, whiten
from lcsa.reliability import spearman_brown

__all__ = ["participant_counts", "fit_participants", "summarise", "run_shard",
           "assemble", "not_run", "merge", "run"]

log = logging.getLogger(__name__)

#: A participant needs at least this many scored targets to be fitted.
MIN_TARGETS = 60
OOV = "<oov>"


def participant_counts(corpus: Corpus, keys, candidates: list, frame) -> dict[str, list[np.ndarray]]:
    """0/1 count vectors per participant on the corpus's frozen candidate sets.

    ``keys`` are the ``(text_id, word_number)`` pairs of the corpus in target
    order and ``candidates`` the frozen candidate lists in the same order,
    both from the build directory (``targets.csv`` and ``candidates.json``); a
    response outside the candidate list goes to the OOV slot when the build
    kept one and is otherwise dropped, exactly as in the pooled build.
    """
    if len(keys) != len(corpus) or len(candidates) != len(corpus):
        raise ValueError(f"{len(keys)} target keys and {len(candidates)} candidate lists for a "
                         f"corpus of {len(corpus)} targets")
    slot = {(int(t), int(w)): i for i, (t, w) in enumerate(keys)}
    index = []
    for i, (words, tg) in enumerate(zip(candidates, corpus)):
        if len(words) != tg.V:
            raise ValueError(f"target {keys[i]}: {len(words)} frozen candidates for V={tg.V}")
        index.append({canonical_word(x): j for j, x in enumerate(words)})
    out: dict[str, list[np.ndarray]] = {}
    for pid, grp in frame.groupby("participant", sort=True):
        counts = [np.zeros(tg.V, dtype=np.float64) for tg in corpus]
        n_hit = 0
        for t, w, resp in zip(grp["text_id"], grp["word_number"], grp["response"]):
            i = slot.get((int(t), int(w)))
            if i is None:
                continue
            j = index[i].get(canonical_word(resp), index[i].get(OOV))
            if j is None:
                continue
            counts[i][j] += 1.0
            n_hit += 1
        if n_hit >= MIN_TARGETS:
            out[str(pid)] = counts
    log.info("%d participants with at least %d scored targets", len(out), MIN_TARGETS)
    return out


def _alignment(corp: Corpus, theta0: np.ndarray, model: Model, directions, kernel) -> float:
    """Cosine between the participant's mismatch and a tilt direction under <., .>_N."""
    num = den_x = den_d = 0.0
    for tgt in corp:
        if tgt.N <= 0:
            continue
        fit_t = evaluate_target(tgt, theta0, model, corp.M, kernel)
        q = np.clip(fit_t.q, 1e-300, None)
        p_emp = (tgt.n + 0.5) / (tgt.N + 0.5 * tgt.V)
        x = whiten(centre(np.log(p_emp) - np.log(q), q), q)
        d = whiten(centre(np.asarray(directions[tgt.index], dtype=np.float64), q), q)
        num += tgt.N * float(x @ d)
        den_x += tgt.N * float(x @ x)
        den_d += tgt.N * float(d @ d)
    return float(num / np.sqrt(den_x * den_d)) if den_x > 0 and den_d > 0 else float("nan")


def fit_participants(corpus: Corpus, counts: dict[str, list[np.ndarray]], theta_pooled: dict,
                     models, prepared: dict | None = None, kernel=POWER, seed: int = 0,
                     participants=None) -> list[dict]:
    """One row per (participant, estimator): pinned and free fits, split halves, alignments."""
    rows = []
    pids = list(counts) if participants is None else [p for p in participants if p in counts]
    dirs = {}
    if prepared is not None:
        for nm in ("N-TOPIC", "N-ORDER"):
            rec = prepared.get("readers", {}).get(nm)
            if rec is not None and "directions" in rec:
                dirs[nm] = rec["directions"]
    for pid in pids:
        corp = corpus.with_counts(counts[pid])
        live = [t.index for t in corp if t.N > 0]
        rng = np.random.default_rng([int(seed), zlib.crc32(str(pid).encode())])
        half = np.zeros(len(corp), dtype=bool)
        pick = rng.permutation(live)
        half[pick[: len(pick) // 2]] = True
        for model in models:
            th0 = np.asarray(theta_pooled[model.name], dtype=np.float64)
            pinned = {i: float(th0[i]) for i in range(1, th0.size)}
            row = {"participant": pid, "estimator": model.name, "n_targets": len(live),
                   "n_responses": float(sum(t.N for t in corp))}
            try:
                fp = fit(corp, model, kernel, fixed=pinned, n_starts=2, seed=seed)
                ff = fit(corp, model, kernel, n_starts=2, seed=seed)
                hA = corpus.with_counts([c if half[i] else np.zeros_like(c)
                                         for i, c in enumerate(counts[pid])])
                hB = corpus.with_counts([c if not half[i] else np.zeros_like(c)
                                         for i, c in enumerate(counts[pid])])
                fA = fit(hA, model, kernel, fixed=pinned, n_starts=1, seed=seed)
                fB = fit(hB, model, kernel, fixed=pinned, n_starts=1, seed=seed)
                row.update({
                    "delta_pinned": float(fp.delta), "at_bound_pinned": bool(fp.at_bound),
                    "delta_free": float(ff.delta), "at_bound_free": bool(ff.at_bound),
                    "converged_free": bool(ff.success),
                    "d_half_pinned": float(d_half_from_delta(fp.delta, kernel=kernel)) if fp.delta > 0 else float("inf"),
                    "delta_half_a": float(fA.delta), "delta_half_b": float(fB.delta),
                    "failed": False,
                })
                th_null = th0.copy(); th_null[0] = 0.0
                for nm, dd in dirs.items():
                    row[f"alignment_{nm}"] = _alignment(corp, th_null, model, dd, kernel)
            except Exception as exc:
                log.debug("participant %s under %s failed: %s", pid, model.name, exc)
                row.update({"failed": True, "error": str(exc)})
            rows.append(row)
    return rows


def summarise(rows: list[dict], external: dict[str, float] | None = None) -> list[dict]:
    """Per estimator: spread, reliability, bound share, and the alignment correlations."""
    from scipy.stats import spearmanr

    out = []
    for est in sorted({r["estimator"] for r in rows}):
        ok = [r for r in rows if r["estimator"] == est and not r.get("failed")]
        if not ok:
            out.append({"estimator": est, "n_participants": 0})
            continue
        dp = np.array([r["delta_pinned"] for r in ok])
        df_ = np.array([r["delta_free"] for r in ok])
        a = np.array([r["delta_half_a"] for r in ok]); b = np.array([r["delta_half_b"] for r in ok])
        r_half = float(spearmanr(a, b).statistic) if len(ok) > 3 else float("nan")
        # The rows carry the half-life under the kernel they were fitted with,
        # which this summary does not otherwise know.
        log_dh = (np.log([r["d_half_pinned"] for r in ok if r["delta_pinned"] > 0])
                  if np.any(dp > 0) else np.array([]))
        row = {
            "estimator": est, "n_participants": len(ok),
            "n_failed": sum(1 for r in rows if r["estimator"] == est and r.get("failed")),
            "median_delta_pinned": float(np.median(dp)),
            "iqr_delta_pinned": float(np.percentile(dp, 75) - np.percentile(dp, 25)),
            "share_at_zero_pinned": float(np.mean([r["at_bound_pinned"] and r["delta_pinned"] <= 1e-4 for r in ok])),
            "sd_log_d_half_pinned": float(np.std(log_dh, ddof=1)) if log_dh.size > 1 else float("nan"),
            "median_delta_free": float(np.median(df_)),
            "split_half_r": r_half,
            "split_half_reliability_sb": float(spearman_brown(r_half)) if np.isfinite(r_half) else float("nan"),
        }
        for nm in ("N-TOPIC", "N-ORDER"):
            key = f"alignment_{nm}"
            vals = [(r["delta_pinned"], r[key]) for r in ok if np.isfinite(r.get(key, np.nan))]
            if len(vals) > 3:
                x, y = zip(*vals)
                row[f"rho_delta_vs_{nm}"] = float(spearmanr(x, y).statistic)
        if external:
            pairs = [(r["delta_pinned"], external[r["participant"]]) for r in ok
                     if r["participant"] in external]
            if len(pairs) > 3:
                x, y = zip(*pairs)
                row["rho_vs_external"] = float(spearmanr(x, y).statistic)
                row["n_external_shared"] = len(pairs)
        out.append(row)
    return out


def run_shard(corpus, counts, theta_pooled, models, out_dir, reps: range, prepared=None,
              kernel=POWER, seed: int = 0) -> list[dict]:
    """Participants ``reps`` (by sorted position) fitted and written as one shard."""
    pids = sorted(counts)[reps.start:reps.stop]
    rows = fit_participants(corpus, counts, theta_pooled, models, prepared, kernel, seed, pids)
    for r in rows:
        r["replicate"] = sorted(counts).index(r["participant"])
    write_shard(out_dir, "e5_participants", reps, rows)
    return rows


def assemble(rows: list[dict], out_dir, external=None) -> dict:
    art = Artifacts(out_dir, "e5")
    art.table("e5_participant_fits", rows)
    summary = summarise(rows, external)
    art.table("e5_summary", summary)
    res = {"summary": summary, "n_rows": len(rows)}
    art.save("e5_summary", res)
    return res


def not_run(out_dir, reason: str) -> dict:
    """The audit's artifact when the per-participant responses are unavailable.

    The raw cloze export is not part of the distributed norms, so this leg can
    have no input at all.  Writing the status is what keeps a missing file
    distinguishable from an audit that ran and found no spread.
    """
    art = Artifacts(out_dir, "e5")
    res = {"status": "not_run", "reason": str(reason), "summary": [], "n_rows": 0}
    art.save("e5_summary", res)
    return res


def merge(out_dir, external=None, n_part: int | None = None) -> dict:
    """``n_part`` is the participant count the array tiled; a shard that never
    ran would otherwise shrink the audit to whoever happened to finish."""
    return assemble(read_shards(out_dir, "e5_participants", n_required=n_part),
                    out_dir, external)


def pooled_theta(corpus: Corpus, models, kernel=POWER, seed: int = 0) -> dict[str, list[float]]:
    """The pooled constrained fit per estimator, the values the participant fits pin."""
    return {m.name: [float(x) for x in fit_constrained(corpus, m, kernel, n_starts=3, seed=seed).theta]
            for m in models}


def run(corpus, counts, models, out_dir, prepared=None, kernel=POWER, seed: int = 0,
        external=None) -> dict:
    theta = pooled_theta(corpus, models, kernel, seed)
    rows = run_shard(corpus, counts, theta, models, out_dir, range(len(counts)), prepared,
                     kernel, seed)
    return assemble(rows, out_dir, external)
