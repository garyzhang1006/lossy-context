"""The frozen registration and the scorecard generated from it.

A pre-registered paper can fail in a way no test suite catches: the text
registers an analysis (a split, a threshold, an interval formula) that the
released code never implemented, and the plan file the text says was hashed is
not in the repository.  The audit of the seed-noise release found exactly that.
This module closes the gap by making the registration a file the pipeline
consumes.  ``lcsa register`` writes the design constants, the thirteen
predictions with their thresholds, the reading rule and the pre-committed gate
branches to ``registration.json`` before any result exists and records its
SHA-256 in a sidecar.  ``lcsa merge`` refuses to run unless that file is present
and unchanged, refuses again if any code constant the registration froze has
drifted, refuses a leg whose artifacts are missing rather than omitting it, and
then writes ``scorecard.csv`` from the frozen thresholds.  A prediction the
code cannot score is an error here, not a sentence in the paper.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["REGISTRATION", "SIDECAR", "build_registration", "registration_hash",
           "write_registration", "load_registration", "check_constants",
           "missing_artifacts", "score", "write_scorecard"]

REGISTRATION = "registration.json"
SIDECAR = "registration.sha256"
VERSION = 1

# Every input a leg must have produced before merge reads it (the summaries
# are what merge writes, so they are not listed).  A leg that is registered and
# absent is a failure, never a silently shorter table.
REQUIRED = {
    "e1": ["e1_summary.json"],
    "e2": ["e2_ladder.json", "shards/e2_coverage_*.json"],
    "e3": ["e3_prepared.json", "e3_human_stage.json",
           "shards/e3_rates_*.json", "shards/e3_contrast_*.json"],
    "e4": ["e4_stage.json", "shards/e4_argmax_*.json"],
    "e5": ["shards/e5_participants_*.json"],
    "e6": ["shards/e6_panel_*.json"],
}


def _jsonable(x):
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    if isinstance(x, (tuple, list)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return x


def build_registration(n_rep: int = 200, n_boot: int = 200, seed: int = 0,
                       readers=None, kernel: str = "power", model: str = "Qwen/Qwen2.5-1.5B",
                       references=(), sweep_references=(), legs=("e1", "e2", "e3", "e4", "e6"),
                       shards=None, frozen_at: str | None = None) -> dict:
    """The registration as a dict, built from the code's own constants.

    The constants are imported rather than retyped so that the file freezes what
    the code will run, and ``check_constants`` later proves the code still
    matches the file.
    """
    from lcsa.experiments.e2_ladder import CEILING_SHARE, COVERAGE_RUNGS
    from lcsa.experiments.e3_nulls import ALPHAS, FLOORS, N_RULE_FLOOR, N_SPLITS, RULE
    from lcsa.experiments.e6_crossed import PANEL_RUNGS, PANEL_TILT
    from lcsa.readers import LADDER_DHALF
    from lcsa.subsetcache import BRIDGING_TARGETS, K_MAX

    readers = list(readers) if readers else ["N0", "N0-PRIME", "N-LEX", "N-TOPIC", "N-ORDER"]
    return {
        "version": VERSION,
        "frozen_at": frozen_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "design": {
            "n_rep": int(n_rep), "n_boot": int(n_boot), "seed": int(seed),
            "kernel": kernel, "primary_model": model,
            "references": list(references), "sweep_references": list(sweep_references),
            "readers": readers, "floors": list(FLOORS),
            "ladder_d_half": _jsonable(LADDER_DHALF),
            "coverage_rungs": _jsonable(COVERAGE_RUNGS),
            "ceiling_share": CEILING_SHARE,
            "panel_rungs": _jsonable(PANEL_RUNGS), "panel_tilt": PANEL_TILT,
            "residual_alphas": _jsonable(ALPHAS), "residual_splits": N_SPLITS,
            "rule_floor_replicates": N_RULE_FLOOR,
            "bridging_k_max": int(K_MAX),
            "bridging_targets": int(BRIDGING_TARGETS),
            "shards": dict(shards or {}),
            "legs": list(legs),
            "replicate_seeding": "replicate b is seeded from (seed, b) alone, so shards tile",
        },
        "gates": {
            "G0": {"branch": "without per-target type counts the multinomial likelihood is "
                             "impossible; the project becomes the reading-time estimator alone "
                             "with E4 as the paper"},
            "G1": {"tflops_floor": 2.5,
                   "branch": "re-price every GPU line, drop the competence readers, freeze the "
                             "reference at Qwen2.5-0.5B before G3 and G4 run, and promote "
                             "Qwen2.5-1.5B to the reference-dependence row of the E4 sweep"},
            "G2": {"tv_floor": 0.02, "frac_floor": 0.80,
                   "branch": "the smallest failing j is the analytic ceiling and prediction 9 reads it"},
            "G3": {"human_js_ratio_floor": 1.5, "gaze_split_half_floor": 0.60,
                   "branch": "contrasts carry an attenuation correction, and below 1.5 the human "
                             "arm is dropped and the paper stands on the synthetic readers"},
            "G4": {"se_cap": 0.09, "rho_cap": 0.15,
                   "branch": "prediction 6 is scored as contrast-interval overlap, not TOST"},
            "G5": {"floor_reject_cap": 0.10,
                   "branch": "predictions 2 to 13 and the reading rule are void"},
            "G6": {"coverage_floor": 0.90, "rungs": [4.0, 8.0]},
            "G7": {"gpu_hours_cap": 10.0, "gpu_hours_budget": 14.0,
                   "branch": "defer the kernel-bridging enumeration, then drop the competence readers"},
        },
        "reading_rule": dict(RULE, estimator="naive", jeffreys_alpha=0.5),
        "predictions": [
            {"id": 1, "leg": "e3", "estimator": "both",
             "statement": "both floors reject at most 0.10 under the one-sided wild cluster score bootstrap",
             "support": {"reject_at_most": 0.10}, "falsify": {"reject_above": 0.15}},
            {"id": 2, "leg": "e3", "reader": "N0-PRIME", "estimator": "naive",
             "statement": "on N0-PRIME the naive LR rejects at least 0.50 while the headline test rejects at most 0.10",
             "support": {"naive_lr_at_least": 0.50, "headline_at_most": 0.10},
             "falsify": {"naive_lr_below": 0.25}},
            {"id": 3, "leg": "e3", "reader": "N-TOPIC", "estimator": "naive",
             "statement": "N-TOPIC rejects at least 0.60 under the naive estimator",
             "support": {"reject_at_least": 0.60}, "falsify": {"reject_at_most": 0.20}},
            {"id": 4, "leg": "e3", "reader": "N-ORDER", "estimator": "naive",
             "statement": "N-ORDER rejects at least 0.60 under the naive estimator",
             "support": {"reject_at_least": 0.60}, "falsify": {"reject_at_most": 0.20}},
            {"id": 5, "leg": "e3", "reader": "N-LEX",
             "statement": "N-LEX rejects at most 0.20 under the repaired estimator and at least 0.40 under the naive one",
             "support": {"repaired_at_most": 0.20, "naive_at_least": 0.40},
             "falsify": {"repaired_at_least": 0.50, "or_ordering_reversed": True}},
            {"id": 6, "leg": "e3", "readers": ["N-TOPIC", "N-ORDER"], "estimator": "naive",
             "statement": "at least one of N-TOPIC, N-ORDER is TOST-equivalent to the human fit on log delta at margin 0.25 if G4 passes, or has an overlapping 95 percent contrast interval otherwise",
             "support": {"tost_margin": 0.25, "overlap_z": 1.96}, "falsify": {"both_separated": True}},
            {"id": 7, "leg": "e4", "estimator": "naive",
             "statement": "a null's spuriously fitted kernel recovers at least 60 percent of the human kernel's held-out reading-time gain",
             "support": {"fraction_at_least": 0.60}, "falsify": {"fraction_below": 0.30}},
            {"id": 8, "leg": "e4",
             "statement": "the context-limitation sweep selects values differing by more than a factor of two across the zero-decay references, or the grid floor for all",
             "support": {"ratio_above": 2.0}, "falsify": {"all_within_ratio": 2.0}},
            {"id": 9, "leg": "e2", "estimator": "naive",
             "statement": "the identification ceiling, as a rate, lies in [12, 30] words",
             "support": {"window": [12.0, 30.0]}, "falsify": {"outside_window": True}},
            {"id": 10, "leg": "e1", "estimator": "naive",
             "statement": "fitted delta under independent deletion and graded truncation agree within 0.25 log units on the 439 short-context targets, the positions at K <= 8",
             "support": {"log_delta_tolerance": 0.25}, "falsify": {"beyond_tolerance": True}},
            {"id": 11, "leg": "e6", "estimator": "naive", "d_half": 8.0,
             "statement": "the N-ORDER tilt shifts the fitted half-life of the d_half=8 reader by at least one rung in at least 0.50 of replicates under the naive estimator",
             "support": {"share_at_least": 0.50}, "falsify": {"share_below": 0.20}},
            {"id": 12, "leg": "e1", "estimator": "n/a",
             "statement": "restoring a passage-initial span of the same length in place of the "
                          "bare truncated prefix moves the candidate distribution by a median "
                          "total-variation gap of at most 0.02, the same floor gate G2 uses to "
                          "call a context moved, over the probed rows where the two spans differ",
             "support": {"median_tv_gap_at_most": 0.02},
             "falsify": {"median_tv_gap_above": 0.05}},
            {"id": 13, "leg": "e3", "estimator": "naive",
             "statement": "the human half-life fitted on an uncapped-depth cache, whose zero-decay "
                          "member conditions on the full passage prefix, agrees with the capped "
                          "value within 0.25 log units, which keeps both fits on one doubling rung",
             "support": {"log_half_life_tolerance": 0.25},
             "falsify": {"beyond_tolerance": True}},
        ],
    }


def registration_hash(reg: dict) -> str:
    """SHA-256 of the canonical JSON; the number the paper prints."""
    canon = json.dumps(reg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def write_registration(out_dir, reg: dict, force: bool = False) -> tuple[Path, str]:
    """Freeze ``reg`` under ``out_dir``; an existing registration is never overwritten silently."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    p = root / REGISTRATION
    if p.exists() and not force:
        raise FileExistsError(
            f"{p} already exists with hash {(root / SIDECAR).read_text().split()[0] if (root / SIDECAR).exists() else '?'}; "
            "a registration is frozen once, pass --force only to start a new run in an empty directory")
    p.write_text(json.dumps(reg, indent=1, sort_keys=True))
    h = registration_hash(reg)
    (root / SIDECAR).write_text(f"{h}  {REGISTRATION}\n")
    return p, h


def load_registration(out_dir) -> tuple[dict, str]:
    """Read and verify the frozen registration; any drift from the sidecar is an error."""
    root = Path(out_dir)
    p, s = root / REGISTRATION, root / SIDECAR
    if not p.exists():
        raise FileNotFoundError(
            f"{p} is missing; run `lcsa register --out {root}` before any leg, or pass "
            "--unregistered to merge an exploratory run whose tables the paper will not quote")
    if not s.exists():
        raise FileNotFoundError(f"{s} is missing; the registration was not frozen by `lcsa register`")
    reg = json.loads(p.read_text())
    want = s.read_text().split()[0]
    have = registration_hash(reg)
    if have != want:
        raise ValueError(f"{p} hashes to {have} but {s} records {want}; the registration was edited after freezing")
    return reg, have


def check_constants(reg: dict) -> list[str]:
    """Code constants that no longer match the frozen design, as messages."""
    fresh = build_registration(
        n_rep=reg["design"]["n_rep"], n_boot=reg["design"]["n_boot"], seed=reg["design"]["seed"],
        readers=reg["design"]["readers"], kernel=reg["design"]["kernel"],
        model=reg["design"]["primary_model"], references=reg["design"]["references"],
        sweep_references=reg["design"]["sweep_references"], legs=reg["design"]["legs"],
        shards=reg["design"]["shards"], frozen_at=reg["frozen_at"])
    out = []
    for key in ("ladder_d_half", "coverage_rungs", "ceiling_share", "panel_rungs", "panel_tilt",
                "residual_alphas", "residual_splits", "rule_floor_replicates", "floors",
                "bridging_k_max", "bridging_targets"):
        if key not in reg["design"]:
            out.append(f"design.{key}: the registration predates this constant, so it froze "
                       f"nothing while the code has {fresh['design'][key]!r}")
        elif fresh["design"][key] != reg["design"][key]:
            out.append(f"design.{key}: code has {fresh['design'][key]!r}, registration froze {reg['design'][key]!r}")
    if fresh["reading_rule"] != reg["reading_rule"]:
        out.append(f"reading_rule: code has {fresh['reading_rule']!r}, registration froze {reg['reading_rule']!r}")
    if fresh["predictions"] != reg["predictions"]:
        out.append("predictions: the thresholds in the code differ from the frozen file")
    return out


def missing_artifacts(out_dir, legs) -> dict[str, list[str]]:
    """Registered legs whose required artifacts are absent, with what is missing."""
    root = Path(out_dir)
    out = {}
    for leg in legs:
        gone = []
        for pat in REQUIRED.get(leg, []):
            hits = list(root.glob(pat)) or list((root / leg).glob(pat))
            if not hits:
                gone.append(pat)
        if gone:
            out[leg] = gone
    return out


# -- scoring ------------------------------------------------------------------

def _summary(root: Path, leg: str):
    for p in (root / f"{leg}_summary.json", root / leg / f"{leg}_summary.json"):
        if p.exists():
            return json.loads(p.read_text())
    return None


def _num(x):
    return float(x) if isinstance(x, (int, float)) and x is not None and math.isfinite(float(x)) else float("nan")


def _rate_row(e3, reader, est):
    return next((r for r in e3["rejection_rates"] if r["reader"] == reader and r["estimator"] == est), None)


def _half_life(arm, est, kernel: str) -> float:
    """Half-life of one human fit, in words; ``nan`` unless the fit is interior.

    The kernel is the arm's own, since a delta fitted under one kernel run
    through the other's formula reports the wrong rung.
    """
    from lcsa.kernels import d_half_from_delta

    row = next((f for f in (arm or {}).get("fits") or [] if f["estimator"] == est), None)
    delta = _num(row.get("delta_hat")) if row else float("nan")
    if not (math.isfinite(delta) and delta > 0):
        return float("nan")
    h = float(d_half_from_delta(delta, kernel))
    return h if math.isfinite(h) and h > 0 else float("nan")


def _verdict(measured: dict, supported, falsified) -> dict:
    if any(isinstance(v, float) and math.isnan(v) for v in measured.values()):
        status = "indeterminate"
    elif falsified:
        status = "falsified"
    elif supported:
        status = "supported"
    else:
        status = "indeterminate"
    return {"status": status, "measured": measured}


def _score_one(pred: dict, summaries: dict, gates: dict) -> dict:
    """Score one prediction against the leg summaries; thresholds come from ``pred``."""
    leg, sup, fal = pred["leg"], pred["support"], pred.get("falsify", {})
    s = summaries.get(leg)
    if s is None:
        return {"status": "missing", "measured": {}}
    if pred["id"] == 1:
        rows = [r for r in s["rejection_rates"] if r["reader"] in ("N0", "N0-PRIME")]
        rates = {f"{r['reader']}/{r['estimator']}": _num(r["reject_cluster_robust"]) for r in rows}
        worst = max(rates.values()) if rates else float("nan")
        return _verdict({"worst_floor_rate": worst, **rates},
                        worst <= sup["reject_at_most"], worst > fal["reject_above"])
    if pred["id"] == 2:
        r = _rate_row(s, pred["reader"], "naive")
        lr = _num(r["reject_naive_LR"]) if r else float("nan")
        hd = _num(r["reject_cluster_robust"]) if r else float("nan")
        return _verdict({"naive_LR_rate": lr, "headline_rate": hd},
                        lr >= sup["naive_lr_at_least"] and hd <= sup["headline_at_most"],
                        lr < fal["naive_lr_below"])
    if pred["id"] in (3, 4):
        r = _rate_row(s, pred["reader"], "naive")
        rate = _num(r["reject_cluster_robust"]) if r else float("nan")
        return _verdict({"rate_naive": rate}, rate >= sup["reject_at_least"], rate <= fal["reject_at_most"])
    if pred["id"] == 5:
        rn, rr = _rate_row(s, pred["reader"], "naive"), _rate_row(s, pred["reader"], "repaired")
        a = _num(rn["reject_cluster_robust"]) if rn else float("nan")
        b = _num(rr["reject_cluster_robust"]) if rr else float("nan")
        return _verdict({"rate_naive": a, "rate_repaired": b},
                        b <= sup["repaired_at_most"] and a >= sup["naive_at_least"],
                        b >= fal["repaired_at_least"] or b > a)
    if pred["id"] == 6:
        human = s.get("human") or {}
        con = (human.get("contrast") or {}).get("contrasts") or {}
        g4_pass = bool(((human.get("g4") or {}).get("passed")))
        meas, hits = {"branch": "tost" if g4_pass else "overlap"}, []
        for nm in pred["readers"]:
            c = con.get(nm)
            if c is None:
                meas[f"{nm}"] = float("nan")
                continue
            if g4_pass:
                ok = bool(c.get("tost_equivalent"))
            else:
                m, se = _num(c.get("mean_log_diff")), _num(c.get("se_log_diff"))
                ok = abs(m) <= sup["overlap_z"] * se if math.isfinite(m) and math.isfinite(se) else float("nan")
            meas[nm] = ok
            hits.append(ok)
        if any(isinstance(h, float) for h in hits) or not hits:
            return {"status": "indeterminate", "measured": meas}
        return _verdict(meas, any(hits), not any(hits))
    if pred["id"] == 7:
        rows = [r for r in s.get("rt_gain", []) if r["reader"] != "human" and r["estimator"] == "naive"]
        best = max((_num(r.get("fraction_of_human_gain")) for r in rows), default=float("nan"))
        return _verdict({"best_null_fraction_of_human_gain": best},
                        best >= sup["fraction_at_least"], best < fal["fraction_below"])
    if pred["id"] == 8:
        p8 = s["prediction_8"]
        ratio = _num(p8.get("max_min_ratio"))
        floor = bool(p8.get("all_at_grid_floor"))
        n = len(p8.get("selected_k", []))
        meas = {"max_min_ratio": ratio, "all_at_grid_floor": floor, "n_references": n}
        if floor:
            return {"status": "supported", "measured": meas}
        if not math.isfinite(ratio):
            return {"status": "indeterminate", "measured": meas}
        return _verdict(meas, ratio > sup["ratio_above"], ratio <= fal["all_within_ratio"])
    if pred["id"] == 9:
        lo, hi = sup["window"]
        g2 = (summaries.get("e1") or {}).get("g2") or {}
        if g2 and not g2.get("passed") and g2["measured"].get("first_failing_j") is not None:
            c = float(g2["measured"]["first_failing_j"])
            meas = {"ceiling_d_half": c, "branch": "G2 analytic ceiling"}
        else:
            row = next((r for r in s["ceiling"] if r["estimator"] == "naive"), None)
            c = _num(row["ceiling_d_half"]) if row else float("nan")
            meas = {"ceiling_d_half": c, "branch": "rate ceiling"}
        return _verdict(meas, lo <= c <= hi, not (lo <= c <= hi))
    if pred["id"] == 10:
        b = s.get("bridging")
        gap = _num(b.get("log_delta_gap")) if b else float("nan")
        return _verdict({"abs_log_delta_gap": abs(gap)}, abs(gap) <= sup["log_delta_tolerance"],
                        abs(gap) > sup["log_delta_tolerance"])
    if pred["id"] == 11:
        row = next((r for r in s["panel"] if r["estimator"] == "naive"
                    and float(r["d_half_true"]) == float(pred["d_half"])), None)
        share = _num(row["share_shift_over_one_rung"]) if row else float("nan")
        return _verdict({"share_shift_over_one_rung": share}, share >= sup["share_at_least"],
                        share < fal["share_below"])
    if pred["id"] == 12:
        # The registered statement restricts to the rows where the two spans
        # differ, which is the probe's truncated median, not its overall one.
        p = s.get("prefix_probe")
        gap = _num(p.get("median_tv_gap_truncated")) if p else float("nan")
        return _verdict({"median_tv_gap_truncated": gap}, gap <= sup["median_tv_gap_at_most"],
                        gap > fal["median_tv_gap_above"])
    if pred["id"] == 13:
        unc = s.get("human_uncapped") or {}
        kernel = unc.get("kernel", "power")
        capped = _half_life(s.get("human"), pred["estimator"], kernel)
        uncapped = _half_life(unc, pred["estimator"], kernel)
        gap = (abs(math.log(uncapped) - math.log(capped))
               if math.isfinite(capped) and math.isfinite(uncapped) else float("nan"))
        # Whether the second arm is attested uncapped or merely deeper decides how
        # much the statement's "uncapped-depth cache" is worth, so the scorecard
        # carries it beside the number rather than leaving it in the leg summary.
        return _verdict({"d_half_capped": capped, "d_half_uncapped": uncapped,
                         "abs_log_half_life_gap": gap,
                         "uncapped_verified": bool(unc.get("uncapped_verified")),
                         "uncapped_attestation": unc.get("attestation")},
                        gap <= sup["log_half_life_tolerance"],
                        gap > sup["log_half_life_tolerance"])
    raise ValueError(f"no scorer for prediction {pred['id']}; the registration lists a prediction the code cannot score")


def score(reg: dict, out_dir) -> dict:
    """Score every registered prediction and the reading rule from the leg summaries."""
    root = Path(out_dir)
    legs = reg["design"]["legs"]
    summaries = {leg: _summary(root, leg) for leg in set(legs) | {"e1", "e2", "e3", "e4", "e6"}}
    e3 = summaries.get("e3") or {}
    g5_failed = bool(e3) and not bool((e3.get("g5") or {}).get("passed", True))
    rows = []
    for pred in reg["predictions"]:
        r = _score_one(pred, summaries, reg["gates"])
        if g5_failed and pred["id"] != 1 and r["status"] != "missing":
            r["status"] = "void"
            r["measured"]["void_reason"] = "G5 failed: a floor rejected above the cap"
        rows.append({"id": pred["id"], "leg": pred["leg"], "statement": pred["statement"],
                     **r, "support": pred["support"], "falsify": pred.get("falsify", {})})
    rule_out = {"status": "missing"}
    rr = (e3.get("human") or {}).get("reading_rule") if e3 else None
    if rr:
        registered = {k: reg["reading_rule"][k] for k in ("floor_signal_cap", "outside_at_least", "absorbable_at_most")}
        if rr.get("thresholds") != registered:
            raise ValueError(f"the reading rule ran with thresholds {rr.get('thresholds')} but {registered} were registered")
        rule_out = {"status": "void" if (g5_failed or not rr.get("floor_check_passed")) else rr["verdict"],
                    "human_fraction_debiased": rr.get("human_fraction_debiased"),
                    "floor_signal_share_median": rr.get("floor_signal_share_median")}
    return {"registration_sha256": registration_hash(reg), "g5_failed": g5_failed,
            "predictions": rows, "reading_rule": rule_out}


def write_scorecard(out_dir, card: dict) -> Path:
    import csv

    root = Path(out_dir)
    (root / "scorecard.json").write_text(json.dumps(card, indent=1, default=str))
    p = root / "scorecard.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prediction", "leg", "status", "measured", "support", "falsify", "statement"])
        for r in card["predictions"]:
            w.writerow([r["id"], r["leg"], r["status"], json.dumps(r["measured"], default=str),
                        json.dumps(r["support"]), json.dumps(r["falsify"]), r["statement"]])
        w.writerow(["reading_rule", "e3", card["reading_rule"]["status"],
                    json.dumps({k: v for k, v in card["reading_rule"].items() if k != "status"}, default=str),
                    "", "", "registered reading rule for the debiased human fraction"])
        w.writerow(["registration_sha256", "", card["registration_sha256"], "", "", "", ""])
    return p
