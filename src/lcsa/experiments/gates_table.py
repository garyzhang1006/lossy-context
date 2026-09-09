"""The registered gate table, assembled from whatever the runs have written.

Each gate's result lives with the leg that measured it; this collects them into
one ``gates.csv`` for the appendix, with a gate that has not run reported as
such rather than omitted.
"""

from __future__ import annotations

import json
from pathlib import Path

from lcsa.experiments import Artifacts, jsonable
from lcsa.gates import g7_compute

__all__ = ["collect"]


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _row(name: str, rec, source: str) -> dict:
    if rec is None:
        return {"gate": name, "passed": None, "source": source, "note": "not run"}
    passed = rec.get("passed")
    row = {"gate": name, "passed": None if passed is None else bool(passed),
           "source": source, "threshold": rec.get("threshold", ""),
           "note": rec.get("notes", "")}
    for k, v in (rec.get("measured") or {}).items():
        row[f"m_{k}"] = v
    return row


def collect(out_dir, build_dir=None, gpu_hours: float | None = None,
            e3_complete: bool = True) -> list[dict]:
    out = Path(out_dir)
    build = Path(build_dir) if build_dir else None
    e1 = _load(out / "e1_summary.json") or {}
    e2 = _load(out / "e2_summary.json") or {}
    e3 = _load(out / "e3_summary.json") or {}
    rel = _load(out / "reliability.json") or {}
    g0 = _load(build / "g0_cache.json") if build else None
    g1 = _load(build / "g1.json") if build else None
    if g0 is not None:
        g0 = {"passed": g0.get("passed"), "measured": {k: v for k, v in g0.items()
                                                       if k not in ("gate", "passed")}}
    if g1 is not None:
        g1 = {"passed": g1.get("passed"), "threshold": f">= {g1.get('threshold', 2.0)} TFLOP/s",
              "measured": {"tflops": g1.get("tflops"), "tokens_forwarded": g1.get("tokens_forwarded")}}
    human = e3.get("human") or {}
    rows = [
        _row("G0", g0, "build/g0_cache.json"),
        _row("G1", g1, "build/g1.json"),
        _row("G2", e1.get("g2"), "e1_summary.json"),
        _row("G3", rel.get("g3"), "reliability.json"),
        _row("G4", human.get("g4"), "e3_summary.json"),
        _row("G5", e3.get("g5"), "e3_summary.json"),
        _row("G6", e2.get("g6"), "e2_summary.json"),
    ]
    if gpu_hours is not None:
        rows.append(_row("G7", jsonable(g7_compute(gpu_hours, e3_complete=e3_complete)),
                         "--gpu-hours"))
    else:
        rows.append(_row("G7", None, "--gpu-hours"))
    art = Artifacts(out_dir, "gates")
    art.table("gates", rows)
    art.save("gates", rows)
    return rows
