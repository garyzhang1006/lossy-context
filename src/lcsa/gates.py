"""The eight pre-registered gates, each with the fallback that keeps a paper standing.

A gate is not a unit test.  It is a measurement taken before the work that
depends on it, with the branch decided in advance, so that a bad number changes
the plan instead of the interpretation.  Each function here returns a record
carrying the measured value, the threshold, the pass flag and the fallback text,
and none of them raises on failure: a gate that fired is a finding to report,
not an exception to swallow.

G0  cache integrity and the Provo join, before any GPU work
G1  sustained throughput of the packed forward, before anything consumes p_ref
G2  displacement sensitivity, the analytic ceiling on recoverable retention
G3  cloze and gaze reliability
G4  precision and clustering, which decides the equivalence claim's form
G5  the floors, run before any substantive null
G6  recovery coverage at ladder rungs 4 and 8
G7  the running compute budget
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

__all__ = ["GateResult", "g0_data_integrity", "g1_throughput", "g2_sensitivity",
           "g3_reliability", "g4_precision", "g5_floors", "g6_coverage",
           "g7_compute", "gate_table"]


@dataclass
class GateResult:
    name: str
    passed: bool
    measured: dict
    threshold: str
    fallback: str
    notes: str = ""

    def as_row(self) -> dict:
        d = {"gate": self.name, "passed": self.passed, "threshold": self.threshold}
        d.update({f"m_{k}": v for k, v in self.measured.items()})
        return d


def g0_data_integrity(raw_norms, provo, corpus=None) -> GateResult:
    """G0: the response file decoded cleanly and the join is on the right words.

    Checks, in order: no U+FFFD replacement characters anywhere in the file, the
    literal response ``"NA"`` survived, empty responses are counted rather than
    silently dropped, and the ``(Text_ID, Word_Number)`` join agrees on the word
    string itself.  The naive join mismatches a few hundred of the 2,687 targets,
    which is exactly the error that would propagate into every cached
    distribution with nothing downstream to catch it.
    """
    import pandas as pd

    text_cols = [c for c in raw_norms.columns if raw_norms[c].dtype == object]
    fffd = int(
        sum(int(raw_norms[c].astype(str).str.contains("�", regex=False).sum())
            for c in text_cols)
    )
    resp_col = next((c for c in raw_norms.columns if c.lower() == "response"), None)
    literal_na = (
        int((raw_norms[resp_col].astype(str).str.strip() == "NA").sum())
        if resp_col else -1
    )
    empty_resp = (
        int((raw_norms[resp_col].astype(str).str.strip() == "").sum())
        if resp_col else -1
    )

    mism = 0
    for r in provo.words.itertuples():
        passage = provo.passages.get(int(r.text_id))
        i = int(r.word_number)  # passages are indexed by word_number
        if passage is None or not (0 <= i < len(passage)):
            mism += 1
        elif str(passage[i]).strip().lower() != str(r.word).strip().lower():
            mism += 1

    counts = provo.words["total_responses"].to_numpy(dtype=float)
    frac25 = float((counts >= 25).mean()) if counts.size else float("nan")
    types = (
        provo.responses.groupby(["text_id", "word_number"]).size().to_numpy()
        if len(provo.responses) else np.array([])
    )

    measured = {
        "replacement_chars": fffd,
        "literal_NA_responses": literal_na,
        "empty_responses": empty_resp,
        "join_mismatches": mism,
        "frac_targets_ge_25_responses": frac25,
        "mean_responses": float(counts.mean()) if counts.size else float("nan"),
        "mean_response_types": float(types.mean()) if types.size else float("nan"),
    }
    if corpus is not None:
        rows_bad = 0
        for t in range(min(len(corpus), 200)):
            P = corpus.target(t).P
            if np.abs(P.sum(axis=1) - 1.0).max() > 1e-6:
                rows_bad += 1
        measured["cache_rows_not_normalised"] = rows_bad
    passed = fffd == 0 and mism == 0 and (frac25 > 0.95 or not np.isfinite(frac25))
    return GateResult(
        "G0", passed, measured,
        "zero replacement characters, zero join mismatches, >95 percent of targets with 25+ responses",
        "without per-target type counts the multinomial likelihood is impossible; "
        "the project becomes the reading-time estimator alone with E4 as the paper",
    )


def g1_throughput(tflops: float, threshold: float = 2.0, reference: str = "Qwen2.5-1.5B") -> GateResult:
    """G1: sustained throughput of the packed forward decides the primary reference."""
    return GateResult(
        "G1", bool(tflops >= threshold),
        {"tflops": float(tflops), "reference": reference},
        f"sustained >= {threshold} TFLOP/s",
        "freeze the reference at Qwen2.5-0.5B before G3 and G4 run, and promote "
        "Qwen2.5-1.5B to the reference-dependence row of the E4 sweep",
    )


def g2_sensitivity(corpus, max_j: int = 16, tv_floor: float = 0.02,
                   frac_floor: float = 0.80) -> GateResult:
    """G2: incremental displacements must stay above a total-variation floor.

    If ``D_j`` vanishes beyond some ``j*``, no estimator can carry information
    about retention past ``j*``, so ``j*`` is an analytic ceiling that
    upper-bounds the empirical one.  A failure here strengthens the paper's
    claim rather than weakening it, which is why the fallback is to report the
    ceiling.
    """
    per_j: dict[int, float] = {}
    first_fail = None
    for j in range(1, max_j + 1):
        tv = []
        for tgt in corpus:
            if tgt.K >= j:
                tv.append(0.5 * float(np.abs(tgt.D[j - 1]).sum()))
        if not tv:
            continue
        frac = float(np.mean(np.asarray(tv) >= tv_floor))
        per_j[j] = frac
        if frac < frac_floor and first_fail is None:
            first_fail = j
    return GateResult(
        "G2", first_fail is None,
        {"frac_above_floor_by_j": per_j, "first_failing_j": first_fail},
        f"at least {frac_floor:.0%} of contexts with ||D_j||_TV >= {tv_floor} for every j <= {max_j}",
        "the smallest failing j becomes a hard analytic ceiling on recoverable "
        "retention depth and is reported as such",
    )


def g3_reliability(js_report: dict, gaze_split_half: float, ratio_floor: float = 1.5,
                   gaze_floor: float = 0.60) -> GateResult:
    """G3: human mismatch must exceed sampling noise, and gaze must be reliable."""
    a = js_report.get("js_vs_reference_debiased", float("nan"))
    b = js_report.get("js_split_half_debiased", js_report.get("js_split_half", float("nan")))
    ratio = float(a / b) if (np.isfinite(a) and np.isfinite(b) and b > 0) else float("nan")
    ok_js = np.isfinite(ratio) and ratio >= ratio_floor
    ok_gaze = np.isfinite(gaze_split_half) and gaze_split_half >= gaze_floor
    return GateResult(
        "G3", bool(ok_js and ok_gaze),
        {"js_ratio": ratio, "js_vs_reference_debiased": a, "js_split_half_debiased": b,
         "gaze_split_half": float(gaze_split_half)},
        f"JS ratio >= {ratio_floor} and gaze split-half >= {gaze_floor}",
        "between 1.5 and 3, proceed with a stated attenuation correction; below 1.5 "
        "the human arm is unusable and the paper stands on the nulls, the ladder and "
        "the ceiling; gaze below the floor makes the reading-time leg descriptive",
    )


def g4_precision(sd_passage_log_delta: float, pair_corr: float, within_rho: float,
                 n_clusters: int = 55, df: int = 4, se_cap: float = 0.09,
                 rho_cap: float = 0.15) -> GateResult:
    """G4: decides in advance whether the equivalence claim survives as equivalence.

    The projected contrast standard error is
    ``SD_ub sqrt(2 (1 - r)) / sqrt(C)`` with ``SD_ub`` the one-sided upper 80
    percent bound on the passage-level standard deviation at ``df`` degrees of
    freedom.  Gating on the quantity the equivalence test actually consumes,
    rather than on split-half reliability of ``delta``, is what keeps an abstract
    sentence off a coin flip.
    """
    from scipy.stats import chi2

    sd = float(sd_passage_log_delta)
    if not np.isfinite(sd) or sd <= 0 or df < 1:
        sd_ub = float("nan")
    else:
        # Upper 80 percent one-sided bound on sigma from a chi-square pivot.
        sd_ub = float(sd * np.sqrt(df / chi2.ppf(0.20, df)))
    r = float(pair_corr)
    se = (
        float(sd_ub * np.sqrt(max(2.0 * (1.0 - r), 0.0)) / np.sqrt(n_clusters))
        if np.isfinite(sd_ub) and np.isfinite(r) else float("nan")
    )
    ok = np.isfinite(se) and se <= se_cap and np.isfinite(within_rho) and within_rho <= rho_cap
    return GateResult(
        "G4", bool(ok),
        {"sd_passage_log_delta": sd, "sd_upper_80": sd_ub, "pair_corr": r,
         "within_passage_rho": float(within_rho), "projected_contrast_se": se},
        f"projected contrast SE <= {se_cap} and within-passage rho <= {rho_cap}",
        "convert the equivalence claim in advance to a reported interval overlap "
        "with the contrast SE printed, and rewrite the abstract sentence before the "
        "September 18 lock rather than after it",
    )


def g5_floors(rates: dict[str, float], cap: float = 0.10) -> GateResult:
    """G5: a floor that rejects above the cap means the pipeline manufactures decay."""
    worst = max((v for v in rates.values() if np.isfinite(v)), default=float("nan"))
    return GateResult(
        "G5", bool(np.isfinite(worst) and worst <= cap),
        {"rates": dict(rates), "worst": float(worst)},
        f"every floor rejects at most {cap} under the cluster-robust test",
        "everything downstream is void until the bug is found; the naive LR rate is "
        "recorded but never gated, since its inflation on the over-dispersed floor is "
        "an expected finding",
    )


def g6_coverage(coverage: dict[float, float], rungs=(4.0, 8.0), floor: float = 0.90,
                miscalibration_floor: float = 0.80) -> GateResult:
    """G6: nominal 95 percent regions must cover at the two shortest-horizon rungs."""
    vals = {k: float(coverage.get(k, float("nan"))) for k in rungs}
    worst = min((v for v in vals.values() if np.isfinite(v)), default=float("nan"))
    return GateResult(
        "G6", bool(np.isfinite(worst) and worst >= floor),
        {"coverage": vals, "worst": worst},
        f"coverage >= {floor} at d_half rungs {list(rungs)}",
        f"coverage below {miscalibration_floor} means the estimator is miscalibrated "
        "rather than unidentified, and miscalibration becomes the finding; the "
        "absorption criterion is unaffected because it is analytic",
    )


def g7_compute(gpu_hours_used: float, budget: float = 14.0, checkpoint: float = 10.0,
               e3_complete: bool = False) -> GateResult:
    """G7: the running compute budget, checked before the nulls finish."""
    breach = (not e3_complete) and gpu_hours_used > checkpoint
    return GateResult(
        "G7", not breach,
        {"gpu_hours_used": float(gpu_hours_used), "budget": budget,
         "checkpoint": checkpoint, "e3_complete": bool(e3_complete)},
        f"at most {checkpoint} audited GPU-hours before the nulls complete, {budget} total",
        "defer the bridging experiment to the post-lock window and then drop the "
        "competence-confound readers, in that order",
    )


def gate_table(results: list[GateResult]):
    """Tidy frame of gate outcomes, for the appendix table."""
    import pandas as pd

    return pd.DataFrame([r.as_row() for r in results])
