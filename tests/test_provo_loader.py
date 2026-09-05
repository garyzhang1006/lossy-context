"""Loading Provo without silently corrupting it, and the gate that checks.

The files here are synthetic but carry the three properties that break a naive
read of the real ones: Latin-1 punctuation, the literal response "NA", and two
arms that do not cover the same words.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lcsa.data.provo import canonical_word, load_provo, read_provo_csv
from lcsa.data.subtlex import load_subtlex
from lcsa.gates import g0_data_integrity

PASSAGES = {
    1: ["The", "sailor", "told", "the", "story", "again"],
    2: ["A", "chemist", "named", "NA", "worked", "late"],
}


def _norms_rows():
    rows = []
    for tid, words in PASSAGES.items():
        for i, w in enumerate(words, start=1):
            # Three response types per target, one of which is the word itself.
            for resp, count in ((w.lower(), 20), ("NA", 3), ("thing", 17)):
                rows.append({
                    "Text_ID": tid,
                    "Word_Number": i,
                    "Word": w,
                    "Word_Content_Or_Function": "Content" if i % 2 else "Function",
                    "Response": resp,
                    "Response_Count": count,
                    "Total_Response_Count": 40,
                })
    return pd.DataFrame(rows)


def _eye_rows(skip=(2, 6)):
    rows = []
    for tid, words in PASSAGES.items():
        for i, w in enumerate(words, start=1):
            if tid == 2 and i in skip:
                continue  # the arms disagree, as they do in the real corpus
            for pid in range(4):
                rows.append({
                    "Text_ID": tid,
                    "Word_Number": i,
                    "Participant_ID": f"p{pid}",
                    "IA_FIRST_RUN_DWELL_TIME": 200 + 10 * pid + 5 * i,
                    "IA_LENGTH": len(w),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def provo_dir(tmp_path):
    norms = _norms_rows()
    # A Latin-1 accented character, which a utf-8 read cannot decode at all.
    norms.loc[0, "Word"] = "Thé"
    norms.to_csv(tmp_path / "Provo_Corpus-Predictability_Norms.csv", index=False,
                 encoding="latin-1")
    _eye_rows().to_csv(tmp_path / "Provo_Corpus-Eyetracking_Data.csv", index=False)
    return tmp_path


def test_latin1_files_are_decoded_not_mangled(provo_dir):
    raw, enc = read_provo_csv(provo_dir / "Provo_Corpus-Predictability_Norms.csv")
    assert enc in ("latin-1", "cp1252")
    assert not raw.astype(str).apply(lambda c: c.str.contains("�")).to_numpy().any()


def test_the_literal_response_NA_survives(provo_dir):
    """Pandas turns "NA" into a missing value by default, deleting a real response."""
    raw, _ = read_provo_csv(provo_dir / "Provo_Corpus-Predictability_Norms.csv")
    assert (raw["Response"].astype(str) == "NA").sum() == 12
    provo = load_provo(provo_dir)
    assert (provo.responses["response"] == "NA").sum() == 12


def test_passages_and_targets_are_reconstructed(provo_dir):
    provo = load_provo(provo_dir)
    assert provo.n_passages == 2
    assert provo.n_targets == 12
    assert provo.n_responses == pytest.approx(12 * 40)
    assert provo.passages[2][3] == "NA"
    assert provo.passages[1][0] == "Thé"
    assert len(provo.passages[1]) == 6


def test_the_two_arms_are_reconciled_to_their_intersection(provo_dir):
    provo = load_provo(provo_dir, require_eye=True)
    assert provo.intersection_size == 10  # twelve targets, two without eye data
    assert provo.gaze is not None
    assert provo.summary()["eyetracked_words"] == 10


def test_missing_eye_file_is_only_fatal_when_required(tmp_path):
    _norms_rows().to_csv(tmp_path / "Provo_Corpus-Predictability_Norms.csv", index=False)
    assert load_provo(tmp_path).gaze is None
    with pytest.raises(FileNotFoundError):
        load_provo(tmp_path, require_eye=True)


def test_a_missing_file_says_where_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="osf.io"):
        load_provo(tmp_path)


def test_missing_columns_are_named(tmp_path):
    pd.DataFrame({"Text_ID": [1], "Word": ["x"]}).to_csv(
        tmp_path / "Provo_Corpus-Predictability_Norms.csv", index=False)
    with pytest.raises(KeyError, match="response"):
        load_provo(tmp_path)


def test_canonical_word_strips_edges_but_keeps_apostrophes():
    assert canonical_word(" The, ") == "the"
    assert canonical_word("don't") == "don't"
    assert canonical_word('"Story."') == "story"


def test_gate_g0_passes_on_a_clean_join(provo_dir):
    raw, _ = read_provo_csv(provo_dir / "Provo_Corpus-Predictability_Norms.csv")
    provo = load_provo(provo_dir)
    g = g0_data_integrity(raw, provo)
    assert g.measured["replacement_chars"] == 0
    assert g.measured["join_mismatches"] == 0
    assert g.measured["literal_NA_responses"] == 12
    assert g.measured["frac_targets_ge_25_responses"] == 1.0
    assert g.passed


def test_gate_g0_catches_a_shifted_join(provo_dir):
    """One misaligned word index is the failure that would poison every cache row."""
    raw, _ = read_provo_csv(provo_dir / "Provo_Corpus-Predictability_Norms.csv")
    provo = load_provo(provo_dir)
    provo.passages[1] = ["WRONG"] + provo.passages[1][1:]
    g = g0_data_integrity(raw, provo)
    assert g.measured["join_mismatches"] >= 1
    assert not g.passed
    assert "reading-time estimator" in g.fallback


def test_subtlex_falls_back_to_a_labelled_uniform(tmp_path):
    """Running without the frequency file is allowed, but the run says so."""
    u = load_subtlex(None)
    assert u.source == "uniform"
    assert np.isfinite(u("anything"))
    v = u.vector(["a", "b", "c"])
    assert v.shape == (3,) and np.all(v > 0)
    assert len(set(v)) == 1  # a stand-in has no frequency information in it


def test_subtlex_reads_a_real_frequency_file(tmp_path):
    p = tmp_path / "subtlex.csv"
    pd.DataFrame({"Word": ["the", "cat", "aardvark"],
                  "FREQcount": [1_000_000, 5_000, 3]}).to_csv(p, index=False)
    u = load_subtlex(p)
    assert u.source != "uniform"
    assert u("the") > u("cat") > u("aardvark")
    # An unseen word gets the add-one floor: finite, and below every seen word.
    assert np.isfinite(u("wordthatdoesnotexist"))
    assert u("wordthatdoesnotexist") <= u("aardvark")
    assert np.all(u.vector(["the", "unseen"]) > 0)
