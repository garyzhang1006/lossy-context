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
# Provo numbers words from 2, because the first word of a passage has no
# context and so no cloze predictability, and the fixtures start there too: a
# fixture numbered from 1 hides an off-by-one that shifts every context string
# by a word against the real corpus.
FIRST_WORD_NUMBER = 2


def _norms_rows():
    rows = []
    for tid, words in PASSAGES.items():
        for i, w in enumerate(words, start=FIRST_WORD_NUMBER):
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


def _eye_rows(skip=(3, 7), shift=()):
    """Eye-tracking rows; passages in ``shift`` carry the next word at each number."""
    rows = []
    for tid, words in PASSAGES.items():
        for i, w in enumerate(words, start=FIRST_WORD_NUMBER):
            if tid == 2 and i in skip:
                continue  # the arms disagree, as they do in the real corpus
            if tid in shift:
                w = words[(i - FIRST_WORD_NUMBER + 1) % len(words)]
            for pid in range(4):
                rows.append({
                    "Text_ID": tid,
                    "Word_Number": i,
                    "Word": w,
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
    # Indexed by word_number, so "NA" is word 5 of passage 2 and the accented
    # first target is word 2 of passage 1; word 1 has no cloze row anywhere.
    assert provo.passages[2][5] == "NA"
    assert provo.passages[1][2] == "Thé"
    assert provo.passages[1][:2] == ["", ""]
    assert len(provo.passages[1]) == FIRST_WORD_NUMBER + len(PASSAGES[1])


def test_a_gap_does_not_shift_the_words_after_it(tmp_path):
    """Three real passages are missing a word, and compacting moved every later one."""
    rows = [r for r in _norms_rows().to_dict("records") if r["Word_Number"] != 4]
    pd.DataFrame(rows).to_csv(
        tmp_path / "Provo_Corpus-Predictability_Norms.csv", index=False)
    provo = load_provo(tmp_path, require_eye=False)
    for r in provo.words.itertuples():
        assert provo.passages[int(r.text_id)][int(r.word_number)] == r.word
    assert provo.passages[1][4] == ""


def test_a_context_skips_the_gap_rather_than_padding_it(tmp_path):
    """A depth of K has to retain K real words even when one number is missing."""
    from lcsa.cache import context_string

    rows = [r for r in _norms_rows().to_dict("records") if r["Word_Number"] != 4]
    pd.DataFrame(rows).to_csv(
        tmp_path / "Provo_Corpus-Predictability_Norms.csv", index=False)
    passage = load_provo(tmp_path, require_eye=False).passages[1]
    # Passage 1 is The sailor told the story again from word 2, without word 4.
    assert context_string(passage, 6, None) == "The sailor the"
    assert context_string(passage, 6, 2) == "sailor the"
    assert context_string(passage, 2, None) == ""


def test_the_two_arms_are_reconciled_to_their_intersection(provo_dir):
    provo = load_provo(provo_dir, require_eye=True)
    assert provo.intersection_size == 10  # twelve targets, two without eye data
    assert provo.gaze is not None
    assert provo.summary()["eyetracked_words"] == 10


def test_an_eye_arm_numbered_differently_is_dropped_not_joined(tmp_path):
    """A shifted eye-tracking passage would regress gaze on the neighbouring word."""
    _norms_rows().to_csv(tmp_path / "Provo_Corpus-Predictability_Norms.csv", index=False)
    _eye_rows(shift=(2,)).to_csv(tmp_path / "Provo_Corpus-Eyetracking_Data.csv", index=False)
    provo = load_provo(tmp_path, require_eye=True)
    # Passage 2 has four eye-tracked numbers after the two skipped ones, and
    # every one of them names the wrong word.
    assert provo.arm_word_mismatches == 4
    assert provo.intersection_size == len(PASSAGES[1])
    assert set(provo.gaze["text_id"]) == {1}
    assert "word" not in provo.gaze.columns
    raw, _ = read_provo_csv(tmp_path / "Provo_Corpus-Predictability_Norms.csv")
    g = g0_data_integrity(raw, provo)
    assert g.measured["arm_word_mismatches"] == 4
    # Four of ten eye-tracked keys is far past the ten percent the gate allows.
    assert not g.passed


def test_an_eye_file_without_a_word_column_cannot_be_checked(tmp_path):
    _norms_rows().to_csv(tmp_path / "Provo_Corpus-Predictability_Norms.csv", index=False)
    _eye_rows().drop(columns=["Word"]).to_csv(
        tmp_path / "Provo_Corpus-Eyetracking_Data.csv", index=False)
    provo = load_provo(tmp_path, require_eye=True)
    assert provo.arm_word_mismatches is None
    assert provo.intersection_size == 10
    assert provo.summary()["arm_word_mismatches"] is None


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
    assert g.measured["arm_word_mismatches"] == 0
    assert g.measured["literal_NA_responses"] == 12
    assert g.measured["frac_targets_ge_25_responses"] == 1.0
    assert g.passed


def test_gate_g0_catches_a_shifted_join(provo_dir):
    """One misaligned word index is the failure that would poison every cache row."""
    raw, _ = read_provo_csv(provo_dir / "Provo_Corpus-Predictability_Norms.csv")
    provo = load_provo(provo_dir)
    provo.passages[1][FIRST_WORD_NUMBER] = "WRONG"
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


def test_a_shifted_arm_drops_the_whole_tail_even_where_words_repeat(tmp_path, monkeypatch):
    """"had had" makes a shifted arm agree at one number by accident, and that
    key would carry a neighbour's reading time if only the disagreeing keys
    were dropped."""
    monkeypatch.setitem(PASSAGES, 3, ["He", "said", "had", "had", "been", "gone"])
    _norms_rows().to_csv(tmp_path / "Provo_Corpus-Predictability_Norms.csv", index=False)
    _eye_rows(shift=(3,)).to_csv(tmp_path / "Provo_Corpus-Eyetracking_Data.csv", index=False)
    provo = load_provo(tmp_path, require_eye=True)
    assert provo.arm_word_mismatches == len(PASSAGES[3])
    assert 3 not in set(provo.gaze["text_id"])


def test_a_blank_word_cell_is_dropped_with_a_warning(tmp_path, caplog):
    """pandas would otherwise write the string "nan" into the passage."""
    norms = _norms_rows()
    norms.loc[norms["Word_Number"] == 4, "Word"] = ""
    norms.to_csv(tmp_path / "Provo_Corpus-Predictability_Norms.csv", index=False)
    _eye_rows().to_csv(tmp_path / "Provo_Corpus-Eyetracking_Data.csv", index=False)
    with caplog.at_level("WARNING"):
        provo = load_provo(tmp_path, require_eye=False)
    assert "blank Word" in caplog.text
    assert provo.passages[1][4] == ""
    assert "nan" not in provo.passages[1]


def test_a_blank_response_cell_loads_and_is_counted_by_the_gate(tmp_path):
    norms = _norms_rows()
    norms.loc[0, "Response"] = ""
    norms.to_csv(tmp_path / "Provo_Corpus-Predictability_Norms.csv", index=False)
    _eye_rows().to_csv(tmp_path / "Provo_Corpus-Eyetracking_Data.csv", index=False)
    provo = load_provo(tmp_path, require_eye=False)
    raw, _ = read_provo_csv(tmp_path / "Provo_Corpus-Predictability_Norms.csv")
    assert g0_data_integrity(raw, provo).measured["empty_responses"] == 1


def test_the_loader_records_the_digest_of_every_file_it_read(provo_dir):
    """The counts in the paper belong to one release, and the digest says which."""
    import hashlib

    provo = load_provo(provo_dir, require_eye=True)
    for name, digest in provo.source_sha256.items():
        on_disk = hashlib.sha256((provo_dir / name).read_bytes()).hexdigest()
        assert digest == on_disk
    assert "Provo_Corpus-Predictability_Norms.csv" in provo.source_sha256
    assert "Provo_Corpus-Eyetracking_Data.csv" in provo.source_sha256
    assert provo.summary()["source_sha256"] == provo.source_sha256


def test_gate_g0_records_the_source_digest_it_audited(provo_dir):
    raw, _ = read_provo_csv(provo_dir / "Provo_Corpus-Predictability_Norms.csv")
    provo = load_provo(provo_dir)
    g = g0_data_integrity(raw, provo)
    assert g.measured["provo_source_sha256"] == provo.source_sha256


# The one disagreement the two real Provo releases carry: the norms tokenise a
# contraction into two numbered words where the eye-tracking file keeps one, so
# the eye numbering runs a word behind the norms for the rest of that passage.
_CONTRACTION_NORMS = ["It", "does", "n't", "matter", "now"]
_CONTRACTION_EYE = ["It", "doesn't", "matter", "now"]
_EYE_CONTENT = {"It": "Function", "doesn't": "Function",
                "matter": "Content", "now": "Content"}


def _contraction_dir(tmp_path, norms_carry_content=True):
    nrows, erows = [], []
    for i, word in enumerate(_CONTRACTION_NORMS, start=FIRST_WORD_NUMBER):
        for resp, count in ((word.lower(), 20), ("thing", 20)):
            row = {"Text_ID": 3, "Word_Number": i, "Word": word,
                   "Response": resp, "Response_Count": count,
                   "Total_Response_Count": 40}
            if norms_carry_content:
                row["Word_Content_Or_Function"] = "Content" if i % 2 else "Function"
            nrows.append(row)
    for i, word in enumerate(_CONTRACTION_EYE, start=FIRST_WORD_NUMBER):
        for pid in range(4):
            erows.append({"Text_ID": 3, "Word_Number": i, "Word": word,
                          "Participant_ID": f"p{pid}",
                          "Word_Content_Or_Function": _EYE_CONTENT[word],
                          "IA_FIRST_RUN_DWELL_TIME": 200 + 10 * pid + 5 * i,
                          "IA_LENGTH": len(word)})
    pd.DataFrame(nrows).to_csv(
        tmp_path / "Provo_Corpus-Predictability_Norms.csv", index=False)
    pd.DataFrame(erows).to_csv(
        tmp_path / "Provo_Corpus-Eyetracking_Data.csv", index=False)
    return tmp_path


def test_a_contraction_the_norms_split_is_renumbered_not_quarantined(tmp_path):
    """Quarantining this shift would throw away most of the real gaze arm."""
    provo = load_provo(_contraction_dir(tmp_path))
    assert provo.arm_word_mismatches == 0
    # "It" keeps 2, "doesn't" takes the number of "does", and the two words
    # after it move up by one; 4 is the "n't" fragment, which has no gaze.
    assert sorted(set(provo.gaze["word_number"])) == [2, 3, 5, 6]
    assert provo.intersection_size == 4
    gaze_at = provo.gaze.groupby("word_number")["gaze"].mean()
    # word_number 5 must carry the times recorded for "matter", not for "n't".
    assert gaze_at[5] == pytest.approx(200 + 15 + 5 * 4)


def test_a_rotation_is_still_quarantined_after_the_split_repair(tmp_path, monkeypatch):
    """The repair must not license any renumbering that merely makes words line up."""
    monkeypatch.setitem(PASSAGES, 3, ["He", "said", "it", "had", "been", "gone"])
    norms = _norms_rows()
    norms.to_csv(tmp_path / "Provo_Corpus-Predictability_Norms.csv", index=False)
    _eye_rows(skip=(), shift=(3,)).to_csv(
        tmp_path / "Provo_Corpus-Eyetracking_Data.csv", index=False)
    provo = load_provo(tmp_path)
    assert provo.arm_word_mismatches == len(PASSAGES[3])
    assert 3 not in set(provo.gaze["text_id"])


def test_is_content_falls_back_to_the_eye_file_when_the_norms_omit_it(tmp_path):
    """Without it every target reads as a function word and one feature is zero."""
    provo = load_provo(_contraction_dir(tmp_path, norms_carry_content=False))
    flags = dict(zip(provo.words["word_number"], provo.words["is_content"]))
    assert flags[2] == 0.0 and flags[3] == 0.0
    assert flags[5] == 1.0 and flags[6] == 1.0
    assert np.isnan(flags[4])  # the fragment the eye file never names


def test_the_norms_content_column_is_kept_when_it_is_present(tmp_path):
    """The eye file is a fallback, so it must not overwrite the norms."""
    provo = load_provo(_contraction_dir(tmp_path))
    flags = dict(zip(provo.words["word_number"], provo.words["is_content"]))
    assert [flags[i] for i in (2, 3, 4, 5, 6)] == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_a_windows_encoded_file_decodes_as_cp1252_not_as_control_characters(tmp_path):
    """Excel writes the smart apostrophe at 0x92, which latin-1 reads as C1."""
    norms = _norms_rows()
    norms.loc[norms["Word"] == "story", "Word"] = "story’s"
    norms.to_csv(tmp_path / "Provo_Corpus-Predictability_Norms.csv", index=False,
                 encoding="cp1252")
    raw, enc = read_provo_csv(tmp_path / "Provo_Corpus-Predictability_Norms.csv")
    assert enc == "cp1252"
    gate = g0_data_integrity(raw, load_provo(tmp_path, require_eye=False))
    assert gate.measured["c1_control_chars"] == 0


def test_a_curly_apostrophe_and_a_typed_one_are_the_same_word_type():
    """Unfolded, the corpus spelling and the typed response split one count in two."""
    assert canonical_word("doesn’t") == canonical_word("doesn't") == "doesn't"
    assert canonical_word("“quoted”") == "quoted"
    assert canonical_word("half—way") == "half-way"


def test_a_content_column_that_says_the_same_thing_everywhere_takes_the_repair(tmp_path):
    """A blank column read as all-function is worth exactly as much as no column."""
    nrows, erows = [], []
    for i, word in enumerate(_CONTRACTION_EYE, start=FIRST_WORD_NUMBER):
        for resp, count in ((word.lower(), 20), ("thing", 20)):
            nrows.append({"Text_ID": 3, "Word_Number": i, "Word": word,
                          "Word_Content_Or_Function": "", "Response": resp,
                          "Response_Count": count, "Total_Response_Count": 40})
        for pid in range(4):
            erows.append({"Text_ID": 3, "Word_Number": i, "Word": word,
                          "Participant_ID": f"p{pid}",
                          "Word_Content_Or_Function": _EYE_CONTENT[word],
                          "IA_FIRST_RUN_DWELL_TIME": 200 + 10 * pid + 5 * i,
                          "IA_LENGTH": len(word)})
    pd.DataFrame(nrows).to_csv(
        tmp_path / "Provo_Corpus-Predictability_Norms.csv", index=False)
    pd.DataFrame(erows).to_csv(
        tmp_path / "Provo_Corpus-Eyetracking_Data.csv", index=False)
    provo = load_provo(tmp_path)
    flags = dict(zip(provo.words["word_number"], provo.words["is_content"]))
    assert [flags[i] for i in (2, 3, 4, 5)] == [0.0, 0.0, 1.0, 1.0]


def test_an_empty_answer_does_not_become_a_vote_for_the_word_nan(tmp_path):
    from lcsa.data.provo import load_cloze_participants

    rows = [{"Participant_ID": "p1", "Text_ID": 1, "Word_Number": 2, "Response": "the"},
            {"Participant_ID": "p1", "Text_ID": 1, "Word_Number": 3, "Response": ""},
            {"Participant_ID": "", "Text_ID": 1, "Word_Number": 4, "Response": "story"}]
    path = tmp_path / "cloze_by_participant.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    frame, _ = load_cloze_participants(path)
    assert list(frame["response"]) == ["the"]
    assert "nan" not in set(frame["response"]) | set(frame["participant"])
