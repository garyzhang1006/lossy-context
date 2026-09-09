"""Loading the Provo corpus without silently corrupting it.

Three details in the published files break a naive ``pd.read_csv`` and each has
bitten this pipeline during development, so each is handled explicitly and
asserted rather than assumed.

1.  Encoding.  The distributed files are not UTF-8; they carry Latin-1
    punctuation.  We try UTF-8, then Latin-1, and record which worked.
2.  The literal response ``"NA"``.  Pandas turns it into a missing value by
    default, which deletes a real cloze response for the word *NA* and, worse,
    shifts nothing visibly.  We pass ``keep_default_na=False`` and treat the
    empty string as the only missing marker.
3.  The two arms disagree.  The predictability norms and the eye-tracking file
    do not cover exactly the same words, so we reconcile to their word-level
    intersection and report the count instead of assuming 2,685 rows line up.

The unit of analysis is the passage (``Text_ID``), and every interval in the
paper prints 55 clusters; items, responses and participants are descriptive
scale only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["ProvoData", "load_provo", "read_provo_csv", "canonical_word",
           "load_cloze_participants"]

_ENCODINGS = ("utf-8", "latin-1", "cp1252")

_NORM_ALIASES = {
    "text_id": ["text_id", "textid"],
    "word_number": ["word_number", "wordnumber", "word_num"],
    "word": ["word"],
    "word_cleaned": ["word_cleaned", "wordcleaned"],
    "response": ["response"],
    "response_count": ["response_count", "responsecount"],
    "total_response_count": ["total_response_count", "totalresponsecount"],
    "text": ["text"],
    "word_content_or_function": ["word_content_or_function"],
    "word_unique_id": ["word_unique_id", "worduniqueid"],
}

# The distributed predictability norms aggregate responses per word, so a
# participant-level file has to come from the raw cloze export on OSF or from
# the authors; its column names are not fixed, hence the aliases.
_PARTICIPANT_ALIASES = {
    "participant": ["participant", "participant_id", "participantid", "subject", "subject_id",
                    "subj", "worker_id", "workerid", "id"],
    "text_id": ["text_id", "textid"],
    "word_number": ["word_number", "wordnumber", "word_num"],
    "response": ["response", "cloze_response", "answer"],
}

_EYE_ALIASES = {
    "text_id": ["text_id", "textid"],
    "word_number": ["word_number", "wordnumber"],
    "participant_id": ["participant_id", "participantid", "subject_id"],
    "gaze": [
        "ia_first_run_dwell_time",
        "ia_dwell_time",
        "gaze_duration",
        "ia_first_run_fixation_duration",
    ],
    "first_fixation": ["ia_first_fixation_duration", "first_fixation_duration"],
    "word_length": ["word_length", "ia_length"],
}


def canonical_word(w: str) -> str:
    """Lowercase and strip surrounding punctuation, keeping internal apostrophes."""
    s = str(w).strip().lower()
    return s.strip(".,;:!?\"'()[]{}<>*")


def read_provo_csv(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Read a Provo csv with the encoding and NA handling the files require."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Provo corpus from "
            "https://osf.io/sjefs/ and point --provo-dir at the folder holding "
            "Provo_Corpus-Predictability_Norms.csv and "
            "Provo_Corpus-Eyetracking_Data.csv"
        )
    last: Exception | None = None
    for enc in _ENCODINGS:
        try:
            df = pd.read_csv(
                path,
                encoding=enc,
                keep_default_na=False,  # keeps the literal response "NA"
                na_values=[""],
                low_memory=False,
            )
            return df, enc
        except UnicodeDecodeError as exc:  # pragma: no cover - file dependent
            last = exc
    raise UnicodeDecodeError(  # pragma: no cover - file dependent
        "provo", b"", 0, 1, f"none of {_ENCODINGS} decoded {path}: {last}"
    )


def _resolve(df: pd.DataFrame, aliases: dict[str, list[str]], required: set[str],
             what: str) -> dict[str, str]:
    lower = {c.lower().strip(): c for c in df.columns}
    out: dict[str, str] = {}
    for key, names in aliases.items():
        for n in names:
            if n in lower:
                out[key] = lower[n]
                break
    missing = required - set(out)
    if missing:
        raise KeyError(
            f"{what} is missing required column(s) {sorted(missing)}; "
            f"found columns {sorted(df.columns)[:20]}"
        )
    return out


@dataclass
class ProvoData:
    """Reconciled Provo arms plus the passage word lists."""

    responses: pd.DataFrame  # text_id, word_number, response, count
    words: pd.DataFrame  # text_id, word_number, word, is_content, total_responses
    passages: dict[int, list[str]]  # text_id -> word list in reading order
    gaze: pd.DataFrame | None  # text_id, word_number, participant_id, gaze
    intersection_size: int
    encoding: str

    @property
    def n_passages(self) -> int:
        return int(self.words["text_id"].nunique())

    @property
    def n_targets(self) -> int:
        return len(self.words)

    @property
    def n_responses(self) -> float:
        return float(self.responses["count"].sum())

    def summary(self) -> dict:
        return {
            "passages": self.n_passages,
            "targets": self.n_targets,
            "responses": self.n_responses,
            "participants_cloze": None,
            "eyetracked_words": (
                int(self.gaze.groupby(["text_id", "word_number"]).ngroups)
                if self.gaze is not None
                else 0
            ),
            "intersection": self.intersection_size,
            "encoding": self.encoding,
        }


def load_provo(
    provo_dir: str | Path,
    norms_name: str = "Provo_Corpus-Predictability_Norms.csv",
    eye_name: str = "Provo_Corpus-Eyetracking_Data.csv",
    require_eye: bool = False,
) -> ProvoData:
    """Load and reconcile the two Provo arms."""
    d = Path(provo_dir)
    norms, enc = read_provo_csv(d / norms_name)
    cols = _resolve(
        norms,
        _NORM_ALIASES,
        {"text_id", "word_number", "word", "response", "response_count"},
        "the predictability norms file",
    )

    n = pd.DataFrame(
        {
            "text_id": pd.to_numeric(norms[cols["text_id"]], errors="coerce"),
            "word_number": pd.to_numeric(norms[cols["word_number"]], errors="coerce"),
            "word": norms[cols["word"]].astype(str),
            "response": norms[cols["response"]].astype(str),
            "count": pd.to_numeric(norms[cols["response_count"]], errors="coerce"),
        }
    )
    if "word_content_or_function" in cols:
        n["is_content"] = (
            norms[cols["word_content_or_function"]].astype(str).str.lower().str[:7]
            == "content"
        ).astype(float)
    else:
        n["is_content"] = np.nan
    n = n.dropna(subset=["text_id", "word_number", "count"])
    n["text_id"] = n["text_id"].astype(int)
    n["word_number"] = n["word_number"].astype(int)
    n = n[n["response"].str.strip() != ""]
    if n.empty:
        raise ValueError("the predictability norms file produced no usable responses")

    # Passage word lists: one row per (text_id, word_number).
    w = (
        n.groupby(["text_id", "word_number"], as_index=False)
        .agg(word=("word", "first"), is_content=("is_content", "first"),
             total_responses=("count", "sum"))
        .sort_values(["text_id", "word_number"])
        .reset_index(drop=True)
    )
    passages: dict[int, list[str]] = {}
    for tid, grp in w.groupby("text_id"):
        g = grp.sort_values("word_number")
        nums = g["word_number"].to_numpy()
        expected = np.arange(nums.min(), nums.max() + 1)
        if not np.array_equal(nums, expected):
            log.warning(
                "passage %s has gaps in word_number (%d words spanning %d..%d); "
                "context strings will use the words present",
                tid, len(nums), nums.min(), nums.max(),
            )
        passages[int(tid)] = [str(x) for x in g["word"].tolist()]

    gaze = None
    eye_path = d / eye_name
    if eye_path.exists():
        eye, _ = read_provo_csv(eye_path)
        ec = _resolve(eye, _EYE_ALIASES, {"text_id", "word_number", "gaze"},
                      "the eye-tracking file")
        gaze = pd.DataFrame(
            {
                "text_id": pd.to_numeric(eye[ec["text_id"]], errors="coerce"),
                "word_number": pd.to_numeric(eye[ec["word_number"]], errors="coerce"),
                "gaze": pd.to_numeric(eye[ec["gaze"]], errors="coerce"),
            }
        )
        gaze["participant_id"] = (
            eye[ec["participant_id"]].astype(str)
            if "participant_id" in ec
            else "unknown"
        )
        gaze = gaze.dropna(subset=["text_id", "word_number", "gaze"])
        gaze["text_id"] = gaze["text_id"].astype(int)
        gaze["word_number"] = gaze["word_number"].astype(int)
        gaze = gaze[gaze["gaze"] > 0]
    elif require_eye:
        raise FileNotFoundError(f"{eye_path} not found and require_eye=True")

    if gaze is not None:
        key_w = set(map(tuple, w[["text_id", "word_number"]].to_numpy()))
        key_g = set(map(tuple, gaze[["text_id", "word_number"]].drop_duplicates().to_numpy()))
        inter = key_w & key_g
        if not inter:
            raise ValueError(
                "the cloze and eye-tracking arms share no (text_id, word_number) key; "
                "check that both files come from the same Provo release"
            )
    else:
        inter = set()

    return ProvoData(
        responses=n[["text_id", "word_number", "response", "count"]].reset_index(drop=True),
        words=w,
        passages=passages,
        gaze=gaze,
        intersection_size=len(inter),
        encoding=enc,
    )


def load_cloze_participants(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Per-participant cloze responses, one row per (participant, target).

    Returns the frame with the four canonical columns ``participant``,
    ``text_id``, ``word_number`` and ``response`` and the encoding that read
    it.  Rows with an empty response are dropped and counted in the log, and a
    participant who answered the same target twice keeps the first answer, so
    that every participant contributes at most one response per target and the
    per-participant count vectors of E5 are 0/1.
    """
    df, enc = read_provo_csv(path)
    cols = _resolve(df, _PARTICIPANT_ALIASES,
                    {"participant", "text_id", "word_number", "response"},
                    "participant cloze file")
    out = pd.DataFrame({
        "participant": df[cols["participant"]].astype(str).str.strip(),
        "text_id": pd.to_numeric(df[cols["text_id"]], errors="coerce"),
        "word_number": pd.to_numeric(df[cols["word_number"]], errors="coerce"),
        "response": df[cols["response"]].astype(str).str.strip(),
    })
    n0 = len(out)
    out = out.dropna(subset=["text_id", "word_number"])
    out = out[out["response"] != ""]
    out["text_id"] = out["text_id"].astype(int)
    out["word_number"] = out["word_number"].astype(int)
    dup = out.duplicated(["participant", "text_id", "word_number"]).sum()
    out = out.drop_duplicates(["participant", "text_id", "word_number"], keep="first")
    log.info("participant cloze file %s: %d rows, %d dropped as empty or unkeyed, "
             "%d duplicate answers dropped, %d participants",
             path, n0, n0 - len(out) - int(dup), int(dup), out["participant"].nunique())
    return out.reset_index(drop=True), enc
