"""SUBTLEX-US unigram frequencies, used as the ``u`` channel of both estimators.

The unigram enters the likelihood through ``(1 - lam) p_delta + lam u``, so its
score direction ``(u - p_delta)/ptilde`` is one of the two nuisance directions
the naive estimator carries.  That direction is neither ``log u`` nor anything a
confound checklist would contain, which is the point of measuring the span
numerically instead of writing it down.

A word absent from SUBTLEX gets the add-one floor rather than zero, since a zero
would make ``u`` an improper mixture component and send the score to infinity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["Unigrams", "load_subtlex"]

_WORD_COLS = ("word", "Word")
_FREQ_COLS = ("FREQcount", "Freq", "frequency", "SUBTLWF", "count")


@dataclass
class Unigrams:
    """Log-probability lookup with an explicit floor for unseen words."""

    logp: dict[str, float]
    floor: float
    total: float
    source: str

    def __call__(self, word: str) -> float:
        return self.logp.get(str(word).strip().lower(), self.floor)

    def vector(self, words) -> np.ndarray:
        """Unigram probabilities for a candidate list, unnormalised."""
        return np.exp(np.array([self(w) for w in words], dtype=np.float64))

    def __len__(self) -> int:
        return len(self.logp)


def load_subtlex(path: str | Path | None, min_count: float = 1.0) -> Unigrams:
    """Load SUBTLEX-US from csv, tsv or xlsx.

    ``path=None`` returns a uniform stand-in so that the estimation pipeline and
    its tests run without the file; every fit that used the stand-in is labelled
    ``uniform`` in its ``source`` field so a run cannot silently report a number
    computed against a placeholder.
    """
    if path is None:
        return Unigrams(logp={}, floor=float(np.log(1e-6)), total=0.0, source="uniform")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. SUBTLEX-US is available from "
            "https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus"
        )
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    else:
        sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
        df = pd.read_csv(p, sep=sep, encoding="latin-1", keep_default_na=False,
                         na_values=[""], low_memory=False)
    wcol = next((c for c in _WORD_COLS if c in df.columns), None)
    fcol = next((c for c in _FREQ_COLS if c in df.columns), None)
    if wcol is None or fcol is None:
        raise KeyError(
            f"{p} has columns {sorted(df.columns)[:15]}; expected a word column from "
            f"{_WORD_COLS} and a frequency column from {_FREQ_COLS}"
        )
    words = df[wcol].astype(str).str.strip().str.lower()
    freq = pd.to_numeric(df[fcol], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    freq = np.clip(freq, 0.0, None) + min_count
    total = float(freq.sum())
    logp = dict(zip(words, np.log(freq / total)))
    floor = float(np.log(min_count / total))
    return Unigrams(logp=logp, floor=floor, total=total, source=str(p))
