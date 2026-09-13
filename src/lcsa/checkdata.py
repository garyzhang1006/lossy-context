"""The whole of G0 without a GPU, so a bad corpus fails on the login node.

``lcsa build`` runs the data-integrity gate, but it runs it inside the job that
holds the GPU, so a corpus the loader cannot reconcile costs an allocation
before it says so.  Everything G0 measures comes off the two csv files and
needs no model, so this module runs the same gate from
:func:`lcsa.gates.g0_data_integrity` on a login node in a few seconds, and adds
the three silent-degradation checks the gate itself does not make: a feature
column that is constant carries no information and nothing downstream raises.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lcsa.subsetcache import BRIDGING_TARGETS


def check_data(
    provo_dir: str | Path,
    subtlex: str | Path | None = None,
    norms_name: str = "Provo_Corpus-Predictability_Norms.csv",
    require_eye: bool = True,
    expect_bridging: int | None = BRIDGING_TARGETS,
) -> tuple[object, dict, dict]:
    """Run G0 and the feature-degradation checks on the real corpus files.

    Returns the :class:`~lcsa.gates.GateResult`, a dict of warnings keyed by a
    short name, and a dict of the counts the submission depends on.  An empty
    warning dict means the build has every channel the registered estimator
    fits, and the ``bridging`` warning is the one that stops a submission,
    since the sub-cache leg raises on that mismatch and the chain is
    ``afterok`` all the way down.
    """
    from lcsa.build import buildable_targets
    from lcsa.data.provo import canonical_word, load_provo, read_provo_csv
    from lcsa.data.subtlex import load_subtlex
    from lcsa.gates import GateResult, g0_data_integrity
    from lcsa.subsetcache import K_MAX, selected_targets

    d = Path(provo_dir)
    # The loader refuses a corpus it cannot reconcile at all, and a login-node
    # check that answers with a traceback is harder to read than one that says
    # what failed, so the refusal is reported as the gate failure it is.
    try:
        provo = load_provo(d, norms_name=norms_name, require_eye=require_eye)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        gate = GateResult(
            "G0", False, {"loader_error": str(exc)},
            "the two arms load and reconcile before any number is measured",
            "no leg of the design can run until the corpus files are fixed",
        )
        return gate, {"loader": str(exc)}, {}
    raw, _ = read_provo_csv(d / norms_name)
    gate = g0_data_integrity(raw, provo)

    warn: dict[str, str] = {}
    words = provo.words
    n_targets = len(words)

    # Both counts are the ones the GPU legs check.  The build admits a target
    # before it scores anything, and the sub-cache leg refuses a selection that
    # disagrees with the frozen design, so counting here turns two allocations
    # into two lines of output.
    admitted = buildable_targets(provo)
    keys = [(int(r.text_id), int(r.word_number)) for r, _, _, _, _ in admitted]
    bridging = selected_targets(provo, keys)
    extra = {
        "targets_the_build_admits": len(keys),
        f"targets_at_K<={K_MAX}": len(bridging),
        "targets_the_registration_froze": expect_bridging,
    }
    if expect_bridging is not None and len(bridging) != int(expect_bridging):
        warn["bridging"] = (
            f"the cap at K <= {K_MAX} selects {len(bridging)} targets and the "
            f"registration froze {int(expect_bridging)}, so lcsa sub-cache will raise "
            "and cancel every job that waits on it"
        )

    known = int(words["is_content"].notna().sum())
    content = int((words["is_content"] == 1.0).sum())
    if known == 0:
        warn["is_content"] = (
            "no arm carries Word_Content_Or_Function, so the is_content feature "
            "is constant and the repaired fit loses a nuisance dimension"
        )
    elif content == 0 or content == known:
        warn["is_content"] = (
            f"is_content is the same value for all {known} targets that carry it, "
            "so that feature column is constant"
        )

    unigrams = load_subtlex(subtlex)
    if unigrams.source == "uniform":
        warn["subtlex"] = (
            "no SUBTLEX file, so log_unigram is a uniform stand-in and the "
            "lexical channel is one feature short"
        )
    else:
        seen = sum(1 for x in words["word"] if canonical_word(x) in unigrams.logp)
        if n_targets and seen / n_targets < 0.90:
            warn["subtlex"] = (
                f"SUBTLEX names only {seen} of {n_targets} targets, so log_unigram "
                "sits at its floor for the rest"
            )

    if provo.gaze is not None:
        covered = provo.intersection_size / n_targets if n_targets else 0.0
        if covered < 0.50:
            warn["gaze"] = (
                f"the gaze arm reaches {provo.intersection_size} of {n_targets} "
                "targets, so E4 is fitted on under half the corpus"
            )
    elif require_eye:
        warn["gaze"] = "no eye-tracking arm, so E4 cannot run"

    return gate, warn, extra


def report(gate, warn: dict, extra: dict | None = None) -> str:
    """One block of text naming every measured value and every warning."""
    lines = [f"G0 {'passed' if gate.passed else 'FAILED'}", f"  threshold: {gate.threshold}"]
    for k, v in gate.measured.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                lines.append(f"  {k}.{kk} = {vv}")
        elif isinstance(v, float) and not np.isfinite(v):
            lines.append(f"  {k} = {v}")
        else:
            lines.append(f"  {k} = {v}")
    for k, v in (extra or {}).items():
        lines.append(f"  {k} = {v}")
    if warn:
        lines.append("warnings (G0 does not fail on these, the fit degrades quietly)")
        lines.extend(f"  {k}: {v}" for k, v in warn.items())
    if not gate.passed:
        lines.append(f"fallback if this cannot be fixed: {gate.fallback}")
    return "\n".join(lines)
