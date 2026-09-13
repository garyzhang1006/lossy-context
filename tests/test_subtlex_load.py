"""Loading SUBTLEX-US without quietly rewriting the frequencies it carries.

The unigram channel is one of the two nuisance directions the estimator has to
span, so a word that takes the count of another word, or falls to the floor
because pandas read it as missing, moves a number nothing downstream checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lcsa.data.subtlex import load_subtlex


def _write(tmp_path, rows, name="subtlex.csv"):
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_case_variants_of_one_word_add_rather_than_overwrite(tmp_path):
    """The file is sorted by frequency, so last-wins hands "will" the count of "Will"."""
    p = _write(tmp_path, {"Word": ["will", "Will", "other"],
                          "FREQcount": [300_000, 8_000, 1_000]})
    u = load_subtlex(p)
    total = 300_000 + 8_000 + 1_000 + 2 * 1.0
    assert u("will") == pytest.approx(float(np.log((308_000 + 1.0) / total)))
    assert len(u.logp) == 2


def test_the_word_null_is_a_word_and_not_a_missing_value(tmp_path):
    p = _write(tmp_path, {"Word": ["null", "none", "NA", "N/A", "the"],
                          "FREQcount": [50, 60, 70, 80, 90]})
    u = load_subtlex(p)
    for w in ("null", "none", "na", "n/a"):
        assert u(w) > u.floor


def test_an_absent_word_takes_the_floor_and_not_a_zero(tmp_path):
    p = _write(tmp_path, {"Word": ["the"], "FREQcount": [90]})
    u = load_subtlex(p)
    assert u("zzzz") == u.floor
    assert np.isfinite(u.floor)


def test_no_file_is_labelled_uniform_so_a_result_cannot_hide_it():
    u = load_subtlex(None)
    assert u.source == "uniform" and len(u) == 0
