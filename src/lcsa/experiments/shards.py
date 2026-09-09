"""Replicate shards: the unit a Slurm array task computes and ``lcsa merge`` sums.

Every replicate loop in E2, E3 and E4 seeds replicate ``b`` from the run seed
and ``b`` alone, so replicates ``[a, b)`` computed in one process are the same
numbers whether or not replicates outside the range were computed alongside
them.  A shard is that half-open range written as one JSON file of per-replicate
rows; merging is concatenation followed by the same summariser the monolithic
run uses, which is what makes the two paths produce identical artifacts.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from lcsa.experiments import jsonable

log = logging.getLogger(__name__)

__all__ = ["rep_indices", "shard_path", "write_shard", "read_shards", "denull"]

SHARD_DIR = "shards"


def rep_indices(n_rep: int, rep_range=None) -> range:
    """Absolute replicate indices for a shard, ``range(n_rep)`` when no range is given."""
    if rep_range is None:
        return range(int(n_rep))
    start, stop = int(rep_range[0]), int(rep_range[1])
    if start < 0 or stop <= start:
        raise ValueError(f"replicate range must satisfy 0 <= start < stop, got {rep_range}")
    return range(start, stop)


def shard_path(out_dir, stem: str, reps: range) -> Path:
    return Path(out_dir) / SHARD_DIR / f"{stem}_{reps.start:06d}-{reps.stop:06d}.json"


def write_shard(out_dir, stem: str, reps: range, rows: list[dict]) -> Path:
    """Write one shard; rows carry their absolute ``replicate`` index."""
    p = shard_path(out_dir, stem, reps)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"stem": stem, "start": reps.start, "stop": reps.stop,
                             "rows": jsonable(rows)}, indent=1))
    log.info("wrote %s (%d rows)", p, len(rows))
    return p


def denull(obj):
    """Undo the one lossy step of JSON: ``None`` in a numeric slot becomes ``nan``."""
    if isinstance(obj, dict):
        return {k: denull(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [denull(v) for v in obj]
    return math.nan if obj is None else obj


def read_shards(out_dir, stem: str) -> list[dict]:
    """Concatenate every shard of ``stem`` and check that the ranges tile.

    Overlapping shards would double-count a replicate and a gap would silently
    report fewer replicates than were registered, so both are errors rather
    than warnings.  Shards may extend past the registered count: the summariser
    then reports the larger number and the first replicates are unchanged.
    """
    d = Path(out_dir) / SHARD_DIR
    files = sorted(d.glob(f"{stem}_*-*.json")) if d.is_dir() else []
    if not files:
        raise FileNotFoundError(f"no shards named {stem}_* under {d}")
    rows, spans = [], []
    for f in files:
        rec = json.loads(f.read_text())
        spans.append((int(rec["start"]), int(rec["stop"]), f.name))
        rows.extend(denull(rec["rows"]))
    spans.sort()
    for (s0, e0, n0), (s1, e1, n1) in zip(spans, spans[1:]):
        if s1 < e0:
            raise ValueError(f"shards {n0} and {n1} overlap on replicates [{s1}, {e0})")
        if s1 > e0:
            raise ValueError(f"replicates [{e0}, {s1}) are missing between {n0} and {n1}")
    if spans[0][0] != 0:
        raise ValueError(f"the first shard {spans[0][2]} starts at {spans[0][0]}, not 0")
    return rows
