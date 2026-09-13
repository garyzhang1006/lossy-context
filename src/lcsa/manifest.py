"""Frozen-output manifest: a SHA-256 per artifact so the paper's numbers can be
checked against the files they came from, and a rerun can prove it changed
nothing.

The manifest also carries the run's diagnostics, so a headline a section of the
paper names is readable beside the hash of the file it came from.  Collection is
by convention rather than by a list of filenames: any JSON artifact whose top
level has a ``diagnostic`` name and a ``summary`` object contributes that
summary under that name, which is how ``prefix_probe.json`` puts its median
total-variation gap here without the manifest knowing what a prefix probe is.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = ["MANIFEST", "write_manifest", "check_manifest"]

MANIFEST = "manifest.json"

# A diagnostic summary is a handful of numbers.  Anything larger is a result
# table that happens to be JSON, and parsing it to find out would cost more than
# hashing it does.
MAX_DIAGNOSTIC_BYTES = 4 << 20


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


def _diagnostics(pairs) -> dict:
    """The ``summary`` block of every self-declaring diagnostic under the root."""
    out: dict = {}
    for rel, p in pairs:
        if p.suffix != ".json" or p.stat().st_size > MAX_DIAGNOSTIC_BYTES:
            continue
        try:
            rec = json.loads(p.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(rec, dict):
            continue
        nm, summary = rec.get("diagnostic"), rec.get("summary")
        if not isinstance(nm, str) or not isinstance(summary, dict):
            continue
        if nm in out:
            log.warning("%s also declares the diagnostic %r, which %s already claimed; "
                        "the manifest keeps the first", rel, nm, out[nm]["source"])
            continue
        out[nm] = {"source": rel, **summary}
    return out


def write_manifest(out_dir, name: str = MANIFEST, diagnostics: dict | None = None) -> dict:
    root = Path(out_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory; nothing to hash")
    pairs = list(_files(root, name))
    entries = {rel: {"sha256": _digest(p), "bytes": p.stat().st_size} for rel, p in pairs}
    diag = _diagnostics(pairs)
    for k, v in (diagnostics or {}).items():
        diag[str(k)] = v
    rec = {"root": root.name, "n_files": len(entries), "diagnostics": diag, "files": entries}
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
