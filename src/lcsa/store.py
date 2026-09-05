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

from lcsa.corpusdata import Corpus

__all__ = ["save_corpus", "load_corpus"]

_FORMAT = 1


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
                  target_slots=[int(s) for s in slots])
