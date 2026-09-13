"""Saving and loading a built corpus, so the GPU pass happens exactly once.

The nested cache is the only expensive artifact in the project.  Everything
downstream, every fit, every replicate, every bootstrap, reads it and never
rebuilds it, so it is written once to a compressed ``.npz`` of flat arrays with
no pickled objects in it.  A file written by one machine loads on another with a
different NumPy, which is the property that matters when the cache is built on a
Kaggle GPU session and analysed on a laptop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lcsa.corpusdata import Corpus, PROB_FLOOR

__all__ = ["save_corpus", "load_corpus", "save_sub_caches", "load_sub_caches"]

_FORMAT = 1
#: Version of the subset-cache file below.  Separate from ``_FORMAT`` because the
#: two artifacts are different shapes and will not move together.
_SUB_FORMAT = 1
#: Written into the subset-cache file so that pointing a loader at the wrong npz
#: fails on the first line with the file named instead of on a missing array.
_SUB_KIND = "sub_caches"


def save_corpus(path: str | Path, corpus: Corpus) -> Path:
    """Write the corpus as flat arrays plus offsets."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Ks, Vs, Pf, nf, uf, gf, ff, wf, slots, cl = [], [], [], [], [], [], [], [], [], []
    for t in corpus:
        Ks.append(t.K)
        Vs.append(t.V)
        Pf.append(np.asarray(t.P, dtype=np.float32).ravel())
        nf.append(np.asarray(t.n, dtype=np.float32))
        uf.append(np.asarray(t.u, dtype=np.float32))
        gf.append(np.asarray(t.g, dtype=np.float32))
        ff.append(np.asarray(t.f, dtype=np.float32).ravel())
        wf.append(np.asarray(t.word_ids, dtype=np.int32))
        slots.append(int(t.target_slot))
        cl.append(int(t.cluster))
    # Written under the same format number on purpose: the build record is two
    # extra arrays a reader that predates it simply never asks for, so a cache
    # written here still loads everywhere the format-1 cache loaded.
    build = {}
    if corpus.max_depth is not None:
        build["max_depth"] = np.array([corpus.max_depth], dtype=np.int32)
    if corpus.n_context is not None:
        build["n_context"] = np.asarray(corpus.n_context, dtype=np.int32)
    if corpus.keys is not None:
        build["keys"] = np.asarray(corpus.keys, dtype=np.int32)
    np.savez_compressed(
        p,
        format=np.array([_FORMAT], dtype=np.int32),
        K=np.asarray(Ks, dtype=np.int32),
        V=np.asarray(Vs, dtype=np.int32),
        M=np.array([corpus.M], dtype=np.int32),
        P=np.concatenate(Pf),
        n=np.concatenate(nf),
        u=np.concatenate(uf),
        g=np.concatenate(gf),
        f=np.concatenate(ff),
        word_ids=np.concatenate(wf),
        target_slot=np.asarray(slots, dtype=np.int32),
        cluster=np.asarray(cl, dtype=np.int64),
        feature_names=np.asarray(list(corpus.feature_names), dtype="U64"),
        **build,
    )
    return p if p.suffix else p.with_suffix(".npz")


def load_corpus(path: str | Path) -> Corpus:
    """Read a corpus written by :func:`save_corpus`."""
    p = Path(path)
    if not p.exists() and p.with_suffix(".npz").exists():
        p = p.with_suffix(".npz")
    with np.load(p, allow_pickle=False) as z:
        fmt = int(z["format"][0])
        if fmt != _FORMAT:
            raise ValueError(
                f"{p} was written in format {fmt}, this build reads format {_FORMAT}"
            )
        K = z["K"].astype(int)
        V = z["V"].astype(int)
        M = int(z["M"][0])
        P, n, u, g, f = z["P"], z["n"], z["u"], z["g"], z["f"]
        wid, slots, cl = z["word_ids"], z["target_slot"], z["cluster"]
        names = [str(x) for x in z["feature_names"]]
        max_depth = int(z["max_depth"][0]) if "max_depth" in z.files else None
        nctx = (z["n_context"].astype(int).tolist() if "n_context" in z.files else None)
        tkeys = (z["keys"].astype(int).tolist() if "keys" in z.files else None)
    P_list, n_list, u_list, g_list, f_list, w_list = [], [], [], [], [], []
    pi = vi = fi = 0
    for t in range(K.size):
        k, v = int(K[t]), int(V[t])
        P_list.append(P[pi: pi + (k + 1) * v].reshape(k + 1, v).astype(np.float64))
        pi += (k + 1) * v
        n_list.append(n[vi: vi + v].astype(np.float64))
        u_list.append(u[vi: vi + v].astype(np.float64))
        g_list.append(g[vi: vi + v].astype(np.float64))
        w_list.append(wid[vi: vi + v].astype(np.int64))
        vi += v
        f_list.append(f[fi: fi + v * M].reshape(v, M).astype(np.float64))
        fi += v * M
    if pi != P.size or vi != n.size or fi != f.size:
        raise ValueError(
            f"{p} has inconsistent offsets: consumed {pi}/{P.size} cache entries, "
            f"{vi}/{n.size} candidate entries, {fi}/{f.size} feature entries"
        )
    return Corpus(P_list, n_list, u_list, f_list, g_list, [int(c) for c in cl],
                  word_ids=w_list, feature_names=names,
                  target_slots=[int(s) for s in slots],
                  n_context=nctx, max_depth=max_depth, keys=tkeys)


def save_sub_caches(path: str | Path, sub_caches: dict) -> Path:
    """Write one ``(2^K, V)`` subset cache per target, keyed by target identity.

    The blocks have different shapes target to target, so they are stored one
    array each rather than concatenated behind offsets, and the key table beside
    them is ``(text_id, word_number)`` rather than a position: a positional index
    would go on resolving after ``build_corpus``'s ``select`` predicate changed
    and would point at a different target every time.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(sub_caches)
    if not keys:
        raise ValueError(f"{p}: there are no subset caches to write")
    blocks, Ks, Vs = {}, [], []
    for i, k in enumerate(keys):
        A = np.asarray(sub_caches[k], dtype=np.float64)
        if A.ndim != 2 or A.shape[0] & (A.shape[0] - 1):
            raise ValueError(
                f"target {tuple(int(x) for x in k)}: a subset cache has shape {A.shape}, "
                "expected (2^K, V)"
            )
        if not np.all(np.isfinite(A)):
            raise ValueError(f"target {tuple(int(x) for x in k)}: non-finite subset cache row")
        Ks.append(int(A.shape[0]).bit_length() - 1)
        Vs.append(int(A.shape[1]))
        # float32 is the precision the nested cache is stored at, so a suffix row
        # here and the depth row it must equal survive the round trip alike.
        blocks[f"sub_{i}"] = A.astype(np.float32)
    np.savez_compressed(
        p,
        format=np.array([_SUB_FORMAT], dtype=np.int32),
        artifact=np.asarray([_SUB_KIND], dtype="U32"),
        text_id=np.asarray([int(k[0]) for k in keys], dtype=np.int32),
        word_number=np.asarray([int(k[1]) for k in keys], dtype=np.int32),
        K=np.asarray(Ks, dtype=np.int32),
        V=np.asarray(Vs, dtype=np.int32),
        **blocks,
    )
    return p if p.suffix else p.with_suffix(".npz")


def load_sub_caches(path: str | Path, corpus: Corpus) -> dict:
    """Read a subset-cache file and key it by the corpus index of each target.

    This is the ``dict[int, np.ndarray]`` that
    :func:`lcsa.experiments.e1_exactness.bridging_report` wants.  The rows are
    clipped and renormalised exactly as :class:`~lcsa.corpusdata.Corpus` treats
    the nested cache, so a suffix row and the depth row it stands for get the
    same arithmetic and not merely the same values.
    """
    p = Path(path)
    if not p.exists() and p.with_suffix(".npz").exists():
        p = p.with_suffix(".npz")
    if corpus.keys is None:
        raise ValueError(
            f"{p} is keyed by (text_id, word_number) but this corpus carries no keys, so "
            "nothing can be resolved against it; rebuild the cache with `lcsa build`, "
            "which records them"
        )
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(corpus.keys)}
    with np.load(p, allow_pickle=False) as z:
        kind = str(z["artifact"][0]) if "artifact" in z.files else None
        if kind != _SUB_KIND:
            raise ValueError(
                f"{p} declares the artifact kind {kind!r}; `lcsa sub-cache` writes "
                f"{_SUB_KIND!r} and this loader reads nothing else"
            )
        fmt = int(z["format"][0])
        if fmt != _SUB_FORMAT:
            raise ValueError(
                f"{p} was written in subset-cache format {fmt}, this build reads "
                f"format {_SUB_FORMAT}"
            )
        tid, wn, K, V = z["text_id"], z["word_number"], z["K"], z["V"]
        out: dict[int, np.ndarray] = {}
        for i in range(tid.size):
            key = (int(tid[i]), int(wn[i]))
            t = index.get(key)
            if t is None:
                raise KeyError(
                    f"{p} holds a subset cache for target {key}, which is not among the "
                    f"{len(corpus)} targets of this corpus; the two came from different "
                    "builds"
                )
            A = z[f"sub_{i}"].astype(np.float64)
            if A.shape != (1 << int(K[i]), int(V[i])):
                raise ValueError(
                    f"{p}: target {key} holds an array of shape {A.shape}, its own key "
                    f"table says {(1 << int(K[i]), int(V[i]))}"
                )
            if A.shape[1] != corpus.target(t).V:
                raise ValueError(
                    f"{p}: target {key} has {A.shape[1]} candidates and the corpus holds "
                    f"{corpus.target(t).V} for it; the two candidate sets were not frozen "
                    "together"
                )
            A = np.clip(A, PROB_FLOOR, None)
            out[t] = A / A.sum(axis=-1, keepdims=True)
    return out
