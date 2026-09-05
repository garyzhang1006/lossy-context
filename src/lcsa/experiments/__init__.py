"""Experiment drivers E1 to E4, plus the artifact convention they share.

Every driver writes JSON and CSV into one output directory and returns the same
dictionary it wrote, so a notebook can inspect the result without re-reading the
file and a re-run overwrites rather than appends.  Nothing here calls the GPU:
the drivers consume a built :class:`~lcsa.corpusdata.Corpus` and are pure CPU,
which is what lets the whole estimation half of the paper run on a laptop.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

__all__ = ["Artifacts", "jsonable"]

log = logging.getLogger(__name__)


def jsonable(obj):
    """Recursively convert numpy scalars, arrays and dataclasses to plain JSON."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return jsonable({k: v for k, v in asdict(obj).items()})
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


class Artifacts:
    """Output directory for one experiment run."""

    def __init__(self, out_dir: str | Path, name: str = "run") -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name

    def save(self, stem: str, obj) -> Path:
        p = self.dir / f"{stem}.json"
        p.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=False))
        log.info("wrote %s", p)
        return p

    def table(self, stem: str, rows) -> Path:
        import pandas as pd

        df = pd.DataFrame([jsonable(r) for r in rows])
        p = self.dir / f"{stem}.csv"
        df.to_csv(p, index=False)
        log.info("wrote %s (%d rows)", p, len(df))
        return p

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"Artifacts({self.dir})"
