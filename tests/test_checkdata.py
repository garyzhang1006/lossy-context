"""The login-node data check must reach the same verdict the GPU job would.

A corpus that fails G0 costs a GPU allocation to discover, so these tests hold
check_data to the gate's own answer and to the three degradations the gate
itself passes over.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lcsa.checkdata import check_data, report

PASSAGE = ["The", "sailor", "told", "the", "story", "again"]
FIRST_WORD_NUMBER = 2


def _write(tmp_path, norms_content=True, eye_content=True, shift=()):
    nrows, erows = [], []
    for tid in (1, 2):
        for i, word in enumerate(PASSAGE, start=FIRST_WORD_NUMBER):
            for resp, count in ((word.lower(), 20), ("thing", 20)):
                row = {"Text_ID": tid, "Word_Number": i, "Word": word, "Response": resp,
                       "Response_Count": count, "Total_Response_Count": 40}
                if norms_content:
                    row["Word_Content_Or_Function"] = "Content" if i % 2 else "Function"
                nrows.append(row)
            seen = (PASSAGE[(i - FIRST_WORD_NUMBER + 1) % len(PASSAGE)]
                    if tid in shift else word)
            for pid in range(4):
                row = {"Text_ID": tid, "Word_Number": i, "Word": seen,
                       "Participant_ID": f"p{pid}",
                       "IA_FIRST_RUN_DWELL_TIME": 200 + 10 * pid, "IA_LENGTH": len(seen)}
                if eye_content:
                    row["Word_Content_Or_Function"] = "Content" if i % 2 else "Function"
                erows.append(row)
    pd.DataFrame(nrows).to_csv(
        tmp_path / "Provo_Corpus-Predictability_Norms.csv", index=False)
    pd.DataFrame(erows).to_csv(
        tmp_path / "Provo_Corpus-Eyetracking_Data.csv", index=False)
    sub = tmp_path / "subtlex.csv"
    pd.DataFrame({"Word": [w.lower() for w in PASSAGE],
                  "FREQcount": [10_000] * len(PASSAGE)}).to_csv(sub, index=False)
    return tmp_path, sub


def test_a_clean_corpus_passes_with_nothing_to_warn_about(tmp_path):
    d, sub = _write(tmp_path)
    gate, warn, extra = check_data(d, subtlex=sub, expect_bridging=None)
    assert gate.passed
    assert warn == {}
    assert extra["targets_the_build_admits"] == len(PASSAGE) * 2 - 2


def test_the_check_fails_where_the_gate_would_fail(tmp_path):
    """A rotated eye arm is the failure that cost an allocation; catch it here."""
    d, sub = _write(tmp_path, shift=(2,))
    gate, warn, extra = check_data(d, subtlex=sub, expect_bridging=None)
    assert not gate.passed
    assert gate.measured["arm_word_mismatch_frac"] > 0.10
    assert gate.fallback in report(gate, warn, extra)


def test_a_corpus_the_loader_refuses_is_reported_not_raised(tmp_path):
    """preflight should print the reason, not a traceback, on the login node."""
    d, sub = _write(tmp_path, shift=(1, 2))
    gate, warn, extra = check_data(d, subtlex=sub, expect_bridging=None)
    assert not gate.passed
    assert "loader" in warn
    assert "numbers its passages differently" in report(gate, warn, extra)


def test_a_missing_content_column_in_both_arms_is_warned_about(tmp_path):
    d, sub = _write(tmp_path, norms_content=False, eye_content=False)
    gate, warn, _ = check_data(d, subtlex=sub, expect_bridging=None)
    assert gate.passed  # G0 says nothing about this
    assert "is_content" in warn


def test_no_subtlex_is_warned_about_rather_than_silently_uniform(tmp_path):
    d, _ = _write(tmp_path)
    gate, warn, _ = check_data(d, subtlex=None, expect_bridging=None)
    assert "subtlex" in warn and "uniform" in warn["subtlex"]


def test_the_report_names_every_measured_value(tmp_path):
    d, sub = _write(tmp_path)
    text = report(*check_data(d, subtlex=sub, expect_bridging=None))
    for key in ("join_mismatches", "arm_word_mismatch_frac", "literal_NA_responses",
                "targets_the_build_admits"):
        assert key in text


def test_a_selection_the_registration_did_not_freeze_is_a_blocking_warning(tmp_path):
    """lcsa sub-cache raises on this and the rest of the chain is afterok."""
    d, sub = _write(tmp_path)
    gate, warn, extra = check_data(d, subtlex=sub, expect_bridging=10_000)
    assert gate.passed
    assert "bridging" in warn and "sub-cache" in warn["bridging"]
    assert extra["targets_the_registration_froze"] == 10_000


def _run_cli(tmp_path, sub, out, *extra):
    from lcsa.cli import build_parser

    args = build_parser().parse_args(
        ["check-data", "--provo-dir", str(tmp_path), "--subtlex", str(sub),
         "--out", str(out), "--expect-bridging", "none", *extra])
    return args.func(args)


def test_the_written_verdict_is_the_exit_status_preflight_greps_for(tmp_path):
    """slurm/preflight.sh reads one key out of this file, so pin the key."""
    import json

    d, sub = _write(tmp_path)
    out = tmp_path / "checkdata"
    assert _run_cli(d, sub, out) == 0
    payload = json.loads((out / "checkdata.json").read_text())
    assert payload["submission_can_proceed"] is True
    assert (out / "checkdata.txt").read_text().strip()

    rotated_dir = tmp_path / "rotated"
    rotated_dir.mkdir()
    rotated, rsub = _write(rotated_dir, shift=(2,))
    rout = tmp_path / "rotated_checkdata"
    assert _run_cli(rotated, rsub, rout) == 1
    assert json.loads((rout / "checkdata.json").read_text())[
        "submission_can_proceed"] is False


def test_preflight_greps_the_key_this_command_writes():
    """The shell string and the json key are edited in different files."""
    from pathlib import Path

    preflight = (Path(__file__).resolve().parents[1] / "slurm" / "preflight.sh").read_text()
    assert '"submission_can_proceed": true' in preflight
