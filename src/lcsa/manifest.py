"""Frozen-output manifest: a SHA-256 per artifact so the paper's numbers can be
checked against the files they came from, and a rerun can prove it changed
nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

__all__ = ["MANIFEST", "write_manifest", "check_manifest"]

MANIFEST = "manifest.json"


def _digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(root: Path, name: str):
    for p in sorted(root.rglob("*")):
        if p.is_file() and not (p.name.startswith("manifest") and p.suffix == ".json"):
            yield p.relative_to(root).as_posix(), p
    # ``name`` is excluded by the pattern above whatever it is called, so a
    # manifest never hashes itself or an earlier manifest.


def write_manifest(out_dir, name: str = MANIFEST) -> dict:
    root = Path(out_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory; nothing to hash")
    entries = {rel: {"sha256": _digest(p), "bytes": p.stat().st_size}
               for rel, p in _files(root, name)}
    rec = {"root": root.name, "n_files": len(entries), "files": entries}
    (root / name).write_text(json.dumps(rec, indent=1))
    return rec


def check_manifest(out_dir, name: str = MANIFEST) -> dict:
    """Compare the directory against ``name``; lists changed, missing and new files."""
    root = Path(out_dir)
    p = root / name
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; run `lcsa manifest --out {root}` first")
    want = json.loads(p.read_text())["files"]
    have = {rel: _digest(q) for rel, q in _files(root, name)}
    changed = sorted(r for r in want if r in have and have[r] != want[r]["sha256"])
    missing = sorted(r for r in want if r not in have)
    new = sorted(r for r in have if r not in want)
    return {"ok": not (changed or missing), "changed": changed, "missing": missing,
            "new": new, "n_checked": len(want)}
