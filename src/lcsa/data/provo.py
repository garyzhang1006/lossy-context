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

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["ProvoData", "load_provo", "read_provo_csv", "canonical_word",
           "load_cloze_participants"]

#: Tried in order, and the loop below advances only on a decode error.  cp1252
#: comes before latin-1 because latin-1 maps every byte to a code point and so
#: never raises, which would make the cp1252 rung unreachable and would turn a
#: corpus exported through Excel, where the smart apostrophe lives at 0x92, into
#: a file of C1 control characters that G0 fails on.
_ENCODINGS = ("utf-8", "cp1252", "latin-1")

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
    "word": ["word", "word_cleaned"],
    "gaze": [
        "ia_first_run_dwell_time",
        "ia_dwell_time",
        "gaze_duration",
        "ia_first_run_fixation_duration",
    ],
    "first_fixation": ["ia_first_fixation_duration", "first_fixation_duration"],
    "word_length": ["word_length", "ia_length"],
    "word_content_or_function": ["word_content_or_function"],
}


#: Typographic punctuation the Provo files carry and cloze typists do not.
_PUNCT_FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
    "\u2026": "...", "\u00a0": " ",
})


def canonical_word(w: str) -> str:
    """Lowercase and strip surrounding punctuation, keeping internal apostrophes.

    Curly quotes and dashes fold to their ASCII spelling first, because the
    corpus carries the curly form and the cloze responses carry the typed one.
    Left unfolded, one word type occupies two rows of the same candidate
    simplex, which splits its response count, misses the unigram table and
    points the target slot at whichever spelling the corpus used.
    """
    s = str(w).translate(_PUNCT_FOLD).strip().lower()
    return s.strip(".,;:!?\"'()[]{}<>*")


def _compare_key(w: str) -> str:
    """The letters and digits of a word, accents folded, for cross-file comparison.

    Punctuation is dropped entirely rather than stripped from the edges because
    the two files can disagree on a curly apostrophe or a hyphen at the same
    word, and the comparison is after a shifted number, not a spelling.
    """
    s = unicodedata.normalize("NFKD", canonical_word(w)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", s)


def _merge_run(seq: list[tuple[int, str]], start: int, target: str) -> int:
    """How many tokens from ``start`` join to spell ``target`` exactly, else 0.

    A run of one is not a merge, so it returns 0 and the caller keeps looking.
    """
    merged, k = "", start
    while k < len(seq) and k - start < 4:
        if not seq[k][1]:
            return 0
        cand = merged + seq[k][1]
        if not target.startswith(cand):
            return 0
        merged, k = cand, k + 1
        if merged == target:
            return k - start if k - start >= 2 else 0
    return 0


def _align_arm(norms: list[tuple[int, str]],
               eye: list[tuple[int, str]]) -> tuple[dict[int, int], int | None]:
    """Map each eye ``word_number`` onto the norms number naming the same word.

    The two Provo releases tokenise contractions differently.  The norms split
    ``doesn't`` into two numbered words where the eye-tracking file keeps one,
    and from that word on the eye numbering runs behind the norms numbering for
    the rest of the passage, which is why a per-key comparison sees a wrong word
    at every later number.  Walking the two lists together recovers those keys,
    and the walk moves the numbering only where the split is provable, since it
    accepts several norms tokens as one eye token when their letters join to
    spell it exactly.  A number missing from one arm is a coverage gap and moves
    nothing.  Anything else stops the walk and the caller quarantines the rest of
    the passage, so an eye file that simply numbers its passages differently is
    still refused rather than re-joined onto its neighbours' reading times.

    Returns the mapping and the eye ``word_number`` where the walk stopped,
    which is ``None`` when the passage aligned to its end.
    """
    out: dict[int, int] = {}
    i = j = off = 0
    while i < len(norms) and j < len(eye):
        nwn, nk = norms[i]
        ewn, ek = eye[j]
        if nwn < ewn + off:      # a target no eye row names
            i += 1
            continue
        if nwn > ewn + off:      # an eye row no target names
            j += 1
            continue
        if nk == ek or not nk or not ek:
            out[ewn] = nwn
            i, j = i + 1, j + 1
            continue
        m = _merge_run(norms, i, ek)
        if m:                    # the norms split one eye token
            out[ewn] = nwn
            off += m - 1
            i, j = i + m, j + 1
            continue
        m = _merge_run(eye, j, nk)
        if m:                    # the eye file split one target; no whole-word time
            off -= m - 1
            i, j = i + 1, j + m
            continue
        return out, ewn
    return out, None


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


def _digest(path: Path) -> str:
    """sha256 of one source file, read in chunks so a large release costs no memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ProvoData:
    """Reconciled Provo arms plus the passage word lists."""

    responses: pd.DataFrame  # text_id, word_number, response, count
    words: pd.DataFrame  # text_id, word_number, word, is_content, total_responses
    passages: dict[int, list[str]]  # text_id -> words indexed by word_number
    gaze: pd.DataFrame | None  # text_id, word_number, participant_id, gaze
    intersection_size: int
    encoding: str
    # Keys the eye-tracking file numbers differently from the norms, found by
    # comparing its word column; None when that file carries no word column.
    arm_word_mismatches: int | None = None
    # sha256 of each source file this record was read from, keyed by file name,
    # so a gate can record which release produced the counts it audits.
    source_sha256: dict[str, str] = field(default_factory=dict)

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
            "arm_word_mismatches": self.arm_word_mismatches,
            "encoding": self.encoding,
            "source_sha256": dict(self.source_sha256),
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
    # After the read, so a missing file still raises the loader's own message.
    digests = {norms_name: _digest(d / norms_name)}
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
            "word": norms[cols["word"]].fillna("").astype(str),
            "response": norms[cols["response"]].fillna("").astype(str),
            "count": pd.to_numeric(norms[cols["response_count"]], errors="coerce"),
        }
    )
    if "word_content_or_function" in cols:
        # A cell that says neither word becomes NaN rather than 0.0, because a
        # blank column read as "every target is a function word" is a constant
        # feature the fit cannot use and no gate looks at, and because the
        # recovery from the eye-tracking arm below keys on the missing value.
        flag = (norms[cols["word_content_or_function"]]
                .fillna("").astype(str).str.strip().str.lower())
        n["is_content"] = np.where(
            flag.str[:7] == "content", 1.0,
            np.where(flag.str[:8] == "function", 0.0, np.nan),
        )
    else:
        log.warning("%s has no Word_Content_Or_Function column; the is_content "
                    "feature is missing for every target", norms_name)
        n["is_content"] = np.nan
    n = n.dropna(subset=["text_id", "word_number", "count"])
    n["text_id"] = n["text_id"].astype(int)
    n["word_number"] = n["word_number"].astype(int)
    n = n[n["response"].str.strip() != ""]
    # A blank Word cell would otherwise enter the passage as the string "nan",
    # be scored as context by the reference model and pass G0's join check,
    # since both sides of that comparison read the same cell.
    blank_word = n["word"].str.strip() == ""
    if blank_word.any():
        log.warning("%d norms rows have a blank Word and are dropped", int(blank_word.sum()))
        n = n[~blank_word]
    if n.empty:
        raise ValueError("the predictability norms file produced no usable responses")
    if (n["word_number"] < 0).any():
        raise ValueError("the predictability norms file has a negative Word_Number, which "
                         "Python list indexing would silently write over another word")

    # Passage word lists: one row per (text_id, word_number).
    w = (
        n.groupby(["text_id", "word_number"], as_index=False)
        .agg(word=("word", "first"), is_content=("is_content", "first"),
             total_responses=("count", "sum"))
        .sort_values(["text_id", "word_number"])
        .reset_index(drop=True)
    )
    # Indexed by word_number, not by position: Provo numbers words from 2,
    # because a passage's first word has no context and so no cloze
    # predictability, and three passages are missing a word in the middle.  A
    # list compacted to the rows present puts the wrong word at every index,
    # which G0 sees as a join mismatch on every target and which would
    # otherwise score every model on a context shifted by one word.  Index 0,
    # word_number 1 and any gap hold the empty string; context_string drops
    # those, so a depth of K still retains K real words.
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
        row = [""] * (int(nums.max()) + 1)
        for num, word in zip(nums, g["word"].tolist()):
            row[int(num)] = str(word)
        passages[int(tid)] = row

    gaze = None
    arm_mism = None
    eye_path = d / eye_name
    if eye_path.exists():
        digests[eye_name] = _digest(eye_path)
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
        if "word" in ec:
            gaze["word"] = eye[ec["word"]].fillna("").astype(str)
        if "word_content_or_function" in ec:
            gaze["content"] = (
                eye[ec["word_content_or_function"]].astype(str).str.lower().str[:7]
                == "content"
            ).astype(float)
        gaze = gaze.dropna(subset=["text_id", "word_number", "gaze"])
        gaze["text_id"] = gaze["text_id"].astype(int)
        gaze["word_number"] = gaze["word_number"].astype(int)
        gaze = gaze[gaze["gaze"] > 0]
        # The two arms are joined on (text_id, word_number) alone, and nothing
        # downstream would notice if the eye-tracking file numbered a passage
        # differently, since E4 would regress gaze on the surprisal of a
        # neighbouring word.  Where the file names the word, each passage is
        # walked against the norms word by word, which both detects a shift and
        # repairs the one shift the two Provo releases actually carry, a
        # contraction the norms split into two numbered words and the
        # eye-tracking file kept as one.  Only a provable split moves a number,
        # so a passage numbered differently for any other reason is quarantined
        # from the first word that will not align to the end of its gaze arm,
        # because a key where the shifted word happens to coincide ("had had",
        # a repeated "the") would otherwise survive with a neighbour's reading
        # time attached.  The comparison folds accents and punctuation away
        # because the two files need not decode under the same encoding, and a
        # key either file cannot name is not evidence.
        if "word" in gaze.columns:
            norm_seq: dict[int, list[tuple[int, str]]] = {}
            for t, k, x in zip(w["text_id"], w["word_number"], w["word"]):
                norm_seq.setdefault(int(t), []).append((int(k), _compare_key(x)))
            norm_key = {(t, k) for t, s in norm_seq.items() for k, _ in s}
            eye_word = (gaze.groupby(["text_id", "word_number"])["word"]
                        .agg(lambda s: next((_compare_key(x) for x in s if _compare_key(x)), "")))
            eye_seq: dict[int, list[tuple[int, str]]] = {}
            for (t, k), x in eye_word.items():
                eye_seq.setdefault(int(t), []).append((int(k), x))

            remap: dict[tuple[int, int], int] = {}
            bad: set[tuple[int, int]] = set()
            stops: dict[int, int] = {}
            drop: set[tuple[int, int]] = set()
            shifted = 0
            for t, seq in sorted(eye_seq.items()):
                ref = sorted(norm_seq.get(t, []))
                if not ref:
                    continue
                seq.sort()
                amap, stop = _align_arm(ref, seq)
                moved = 0
                for ewn, nwn in amap.items():
                    remap[(t, ewn)] = nwn
                    moved += nwn != ewn
                shifted += moved
                if stop is not None:
                    stops[t] = stop
                    bad |= {(t, k) for k, _ in seq if k >= stop and (t, k) in norm_key}
                # Once a passage is renumbered, a key the walk never paired has
                # to go with it: that key names a word the norms do not, or half
                # of one the eye file split, so keeping its own number would put
                # a neighbour's reading time on a target.  A passage that moved
                # nothing keeps every key it had.
                if moved:
                    drop |= {(t, k) for k, _ in seq if k not in amap}
            drop |= bad
            arm_mism = len(bad)
            if shifted:
                log.info(
                    "%d eye-tracking keys were renumbered onto the norms numbering, "
                    "which the two releases split differently at contractions", shifted,
                )
            if bad:
                log.warning(
                    "%d passages carry words the eye-tracking file and the norms "
                    "cannot align (first at %s); %d (text_id, word_number) keys are "
                    "dropped from the gaze arm from that word to the end of the passage",
                    len(stops), sorted(stops.items())[:5], arm_mism,
                )
            keys = list(zip(gaze["text_id"], gaze["word_number"]))
            if drop:
                keep = [k not in drop for k in keys]
                gaze = gaze[keep]
                keys = [k for k, ok in zip(keys, keep) if ok]
            if shifted:
                gaze = gaze.assign(word_number=[remap.get(k, k[1]) for k in keys])
            gaze = gaze.drop(columns=["word"])
        # The 2018 norms release does not always carry the content/function
        # column the eye-tracking release does, and without it every target
        # reads as a function word, which silently zeroes one of the four
        # lexical features the repaired estimator fits.  A column that is
        # present and says the same thing for every target is worth exactly as
        # much as an absent one, so both take the repair.  This sits outside
        # the alignment block above because an eye file can carry the flag
        # without carrying the word column the walk needs.
        if "content" in gaze.columns:
            uninformative = (w["is_content"].isna().all()
                             or w["is_content"].nunique(dropna=True) <= 1)
            if uninformative and gaze["content"].notna().any():
                flag = (gaze.dropna(subset=["content"])
                        .groupby(["text_id", "word_number"])["content"].first())
                w = w.assign(is_content=[
                    flag.get((int(t), int(k)), np.nan)
                    for t, k in zip(w["text_id"], w["word_number"])
                ])
                log.info("is_content was recovered for %d of %d targets from %s",
                         int(w["is_content"].notna().sum()), len(w), eye_name)
            gaze = gaze.drop(columns=["content"])
    elif require_eye:
        raise FileNotFoundError(f"{eye_path} not found and require_eye=True")

    if gaze is not None:
        key_w = set(map(tuple, w[["text_id", "word_number"]].to_numpy()))
        key_g = set(map(tuple, gaze[["text_id", "word_number"]].drop_duplicates().to_numpy()))
        inter = key_w & key_g
        if not inter:
            raise ValueError(
                "the cloze and eye-tracking arms share no (text_id, word_number) key"
                + (f" after {arm_mism} keys were dropped for naming a different word "
                   "in each file; the eye-tracking file numbers its passages differently"
                   if arm_mism else "; check that both files come from the same Provo release")
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
        arm_word_mismatches=arm_mism,
        source_sha256=digests,
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
        "participant": df[cols["participant"]].fillna("").astype(str).str.strip(),
        "text_id": pd.to_numeric(df[cols["text_id"]], errors="coerce"),
        "word_number": pd.to_numeric(df[cols["word_number"]], errors="coerce"),
        "response": df[cols["response"]].fillna("").astype(str).str.strip(),
    })
    n0 = len(out)
    out = out.dropna(subset=["text_id", "word_number"])
    # Without the fillna above, an empty cell reaches astype(str) as NaN and
    # becomes the four-character string "nan", which survives this filter as a
    # vote for a word nobody typed and as a participant nobody is.
    out = out[(out["response"] != "") & (out["participant"] != "")]
    out["text_id"] = out["text_id"].astype(int)
    out["word_number"] = out["word_number"].astype(int)
    dup = out.duplicated(["participant", "text_id", "word_number"]).sum()
    out = out.drop_duplicates(["participant", "text_id", "word_number"], keep="first")
    log.info("participant cloze file %s: %d rows, %d dropped as empty or unkeyed, "
             "%d duplicate answers dropped, %d participants",
             path, n0, n0 - len(out) - int(dup), int(dup), out["participant"].nunique())
    return out.reset_index(drop=True), enc
