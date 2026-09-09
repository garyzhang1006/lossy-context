"""E6, the crossed panel: real decay and a registered mismatch in the same reader.

E2 asks whether decay is recovered when nothing else is wrong, and E3 asks
whether a mismatch with no decay in it is read as decay.  Neither says what
the estimator does when both are present, which is the regime the human data
are most likely in.  This leg generates counts by graded truncation at a
known ``d_half`` from the primary cache, and then tilts every depth's row by
the N-ORDER direction at the alpha E3 calibrated, so the reader has a true
half-life and a mismatch of human size at once.  The plain arm at the same
``d_half`` shares its retention draws with the tilted arm.

What is reported per rung and estimator is the median fitted half-life in
each arm, the paired shift between arms, and coverage of the true value in
the tilted arm; a mismatch that moves the fitted half-life by more than the
rung spacing in the window where the human estimate lands would mean the
human number cannot be read as a half-life even if E2 recovers it cleanly.
"""

from __future__ import annotations

import logging

import numpy as np

from lcsa.corpusdata import Corpus
from lcsa.experiments import Artifacts
from lcsa.experiments.e2_ladder import fit_rung
from lcsa.experiments.shards import denull, read_shards, write_shard
from lcsa.kernels import POWER, d_half_from_delta
from lcsa.likelihood import Model
from lcsa.readers import reader_ladder

__all__ = ["PANEL_RUNGS", "panel_replicates", "summarise_panel", "run_shard", "assemble",
           "merge", "run"]

log = logging.getLogger(__name__)

#: True half-lives crossed with the mismatch; they bracket the registered window.
PANEL_RUNGS = (4.0, 8.0, 16.0)
#: The mismatch crossed with decay.  N-ORDER is the sharper of the two
#: substantive nulls, and a topic tilt is available through the same code.
PANEL_TILT = "N-ORDER"


def _directions(prepared: dict, name: str) -> tuple[list[np.ndarray], float]:
    rec = prepared["readers"].get(name)
    if rec is None or "directions" not in rec:
        raise KeyError(f"the prepared E3 stage carries no {name} directions; run "
                       f"`lcsa e3 --stage prepare --nulls order` first")
    return rec["directions"], float(rec["alpha"])


def panel_replicates(
    corpus: Corpus,
    theta_nuisance: np.ndarray,
    prepared: dict,
    model: Model,
    reps=range(200),
    rungs=PANEL_RUNGS,
    kernel=POWER,
    seed: int = 0,
    tilt_name: str = PANEL_TILT,
    profile: bool = True,
) -> list[dict]:
    """One row per (rung, arm, replicate); arms are ``plain`` and ``tilted``."""
    dirs, alpha = _directions(prepared, tilt_name)
    rows = []
    for d in rungs:
        for b in reps:
            rep_seed = int(seed) + 1_000_003 * int(b) + int(round(d))
            for arm, tilt in (("plain", None), ("tilted", (dirs, alpha))):
                try:
                    counts = reader_ladder(corpus, float(d), theta_nuisance, model,
                                           seed=rep_seed, kernel=kernel, tilt=tilt)
                    row = fit_rung(corpus, corpus, float(d), theta_nuisance, model, kernel,
                                   seed=rep_seed, grid="local", profile=profile,
                                   gen_model=model, counts=counts)
                    row.update({"arm": arm, "replicate": int(b), "failed": False,
                                "tilt": tilt_name, "alpha": alpha if tilt else 0.0})
                except Exception as exc:
                    log.debug("panel replicate %d rung %s arm %s failed: %s", b, d, arm, exc)
                    row = {"estimator": model.name, "d_half_true": float(d), "arm": arm,
                           "replicate": int(b), "failed": True, "error": str(exc)}
                rows.append(row)
    return rows


def summarise_panel(rows: list[dict]) -> list[dict]:
    """Per (estimator, rung): both arms' medians, the paired shift, tilted coverage."""
    groups: dict[tuple, dict[str, dict[int, dict]]] = {}
    for r in rows:
        if r.get("failed"):
            continue
        key = (str(r["estimator"]), float(r["d_half_true"]))
        groups.setdefault(key, {}).setdefault(r["arm"], {})[int(r["replicate"])] = r
    out = []
    for (est, d), arms in groups.items():
        plain, tilted = arms.get("plain", {}), arms.get("tilted", {})
        both = sorted(set(plain) & set(tilted))
        dh_p = np.array([plain[b]["d_half_hat"] for b in both], dtype=np.float64)
        dh_t = np.array([tilted[b]["d_half_hat"] for b in both], dtype=np.float64)
        fin = np.isfinite(dh_p) & np.isfinite(dh_t)
        shift = np.log2(dh_t[fin]) - np.log2(dh_p[fin])
        cov_t = [bool(tilted[b].get("covers_truth")) for b in tilted]
        cov_p = [bool(plain[b].get("covers_truth")) for b in plain]
        unb_t = [bool(tilted[b].get("unbounded_hi")) for b in tilted]
        out.append({
            "estimator": est, "d_half_true": d,
            "n_pairs": int(len(both)), "n_pairs_finite": int(fin.sum()),
            "median_d_half_plain": float(np.nanmedian(dh_p)) if dh_p.size else float("nan"),
            "median_d_half_tilted": float(np.nanmedian(dh_t)) if dh_t.size else float("nan"),
            "median_log2_shift": float(np.median(shift)) if shift.size else float("nan"),
            "log2_shift_q25": float(np.percentile(shift, 25)) if shift.size else float("nan"),
            "log2_shift_q75": float(np.percentile(shift, 75)) if shift.size else float("nan"),
            "share_shift_over_one_rung": float(np.mean(np.abs(shift) > 1.0)) if shift.size else float("nan"),
            "coverage_plain": float(np.mean(cov_p)) if cov_p else float("nan"),
            "coverage_tilted": float(np.mean(cov_t)) if cov_t else float("nan"),
            "frac_unbounded_tilted": float(np.mean(unb_t)) if unb_t else float("nan"),
        })
    return out


def run_shard(corpus, theta_nuisance, prepared, models, out_dir, reps: range,
              kernel=POWER, seed: int = 0, rungs=PANEL_RUNGS) -> list[dict]:
    rows = []
    for m in models:
        rows += panel_replicates(corpus, theta_nuisance, prepared, m, reps, rungs, kernel, seed)
    write_shard(out_dir, "e6_panel", reps, rows)
    return rows


def assemble(rows: list[dict], out_dir) -> dict:
    art = Artifacts(out_dir, "e6")
    summary = summarise_panel(rows)
    art.table("e6_panel", summary)
    res = {"panel": summary, "n_rows": len(rows)}
    art.save("e6_summary", res)
    return res


def merge(out_dir) -> dict:
    return assemble(denull(read_shards(out_dir, "e6_panel")), out_dir)


def run(corpus, theta_nuisance, prepared, models, out_dir, kernel=POWER, n_rep: int = 200,
        seed: int = 0, rungs=PANEL_RUNGS) -> dict:
    rows = run_shard(corpus, theta_nuisance, prepared, models, out_dir, range(n_rep),
                     kernel, seed, rungs)
    return assemble(rows, out_dir)
