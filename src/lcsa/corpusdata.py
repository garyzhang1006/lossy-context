"""Ragged in-memory container for the nested-ablation cache and the responses.

One *target* is one cloze position.  For target ``t`` we hold

    P_t   (K_t + 1, V_t)  reference distributions at ablation depths 0..K_t
    n_t   (V_t,)          response counts on the candidate set
    u_t   (V_t,)          SUBTLEX-US unigram probabilities, renormalised on V_t
    f_t   (V_t, M)        lexical production features
    g_t   (V_t,)          prior-occurrence indicator
    cluster_t             passage index, the independent unit

Row ``j`` of ``P_t`` is the reference conditioned on the last ``j`` words, so
``P[0]`` is the empty-context distribution and ``P[K]`` is the full context.
This is the order Proposition 1 uses: at ``delta = 0`` the truncation weights
put all their mass on ``j = K``, which is full retention, and at large ``delta``
they concentrate on ``j = 0``.

Storage is one flat float32 buffer with per-target offsets, which keeps 2,687
targets at roughly 40 MB and avoids 2,687 separate small allocations.  Views
handed out by :meth:`Corpus.target` are float64 copies, since every downstream
routine wants float64 arithmetic and the copies are at most 120 by 33.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

__all__ = ["Target", "Corpus", "PROB_FLOOR"]

#: Cache rows are clamped here before normalisation.  A depth-ablated reference
#: can assign an exactly-zero fp16 probability to a candidate, and a single zero
#: would send the log-likelihood to -inf for a word somebody actually produced.
PROB_FLOOR = 1e-12


@dataclass(frozen=True)
class Target:
    """One cloze position, fully materialised in float64."""

    index: int
    cluster: int
    P: np.ndarray  # (K+1, V)
    n: np.ndarray  # (V,)
    u: np.ndarray  # (V,)
    f: np.ndarray  # (V, M)
    g: np.ndarray  # (V,)
    D: np.ndarray  # (K, V) incremental displacements, cached
    word_ids: np.ndarray  # (V,) global candidate ids, for reporting only
    target_slot: int = -1  # index of the corpus word in the candidate set, -1 if absent

    @property
    def K(self) -> int:
        return self.P.shape[0] - 1

    @property
    def V(self) -> int:
        return self.P.shape[1]

    @property
    def N(self) -> float:
        return float(self.n.sum())


def _normalise_rows(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, PROB_FLOOR, None)
    return P / P.sum(axis=-1, keepdims=True)


class Corpus:
    """Ragged collection of :class:`Target` objects sharing one flat buffer."""

    def __init__(
        self,
        P_list: Sequence[np.ndarray],
        n_list: Sequence[np.ndarray],
        u_list: Sequence[np.ndarray],
        f_list: Sequence[np.ndarray],
        g_list: Sequence[np.ndarray],
        clusters: Sequence[int],
        word_ids: Sequence[np.ndarray] | None = None,
        feature_names: Sequence[str] | None = None,
        target_slots: Sequence[int] | None = None,
    ) -> None:
        T = len(P_list)
        if not (len(n_list) == len(u_list) == len(f_list) == len(g_list) == len(clusters) == T):
            raise ValueError(
                "ragged inputs disagree on length: "
                f"P={len(P_list)} n={len(n_list)} u={len(u_list)} "
                f"f={len(f_list)} g={len(g_list)} clusters={len(clusters)}"
            )
        if T == 0:
            raise ValueError("Corpus needs at least one target")

        self.n_targets = T
        self.clusters = np.asarray(clusters, dtype=np.int64)
        uniq = np.unique(self.clusters)
        self.cluster_ids = uniq
        self.n_clusters = int(uniq.size)
        # Dense 0..C-1 relabelling, so downstream code can use bincount safely.
        remap = {int(c): i for i, c in enumerate(uniq)}
        self.cluster_index = np.array([remap[int(c)] for c in self.clusters], dtype=np.int64)

        self.M = int(np.asarray(f_list[0]).shape[1]) if np.asarray(f_list[0]).ndim == 2 else 0
        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"f{i}" for i in range(self.M)]
        )
        if len(self.feature_names) != self.M:
            raise ValueError(
                f"{len(self.feature_names)} feature names for {self.M} feature columns"
            )

        self._K = np.empty(T, dtype=np.int64)
        self._V = np.empty(T, dtype=np.int64)
        for t in range(T):
            P = np.asarray(P_list[t])
            if P.ndim != 2:
                raise ValueError(f"target {t}: P must be 2-D, got shape {P.shape}")
            V = P.shape[1]
            for name, arr, want in (
                ("n", n_list[t], (V,)),
                ("u", u_list[t], (V,)),
                ("g", g_list[t], (V,)),
            ):
                a = np.asarray(arr)
                if a.shape != want:
                    raise ValueError(
                        f"target {t}: {name} has shape {a.shape}, expected {want}"
                    )
            fa = np.asarray(f_list[t])
            if fa.shape != (V, self.M):
                raise ValueError(
                    f"target {t}: f has shape {fa.shape}, expected {(V, self.M)}"
                )
            self._K[t] = P.shape[0] - 1
            self._V[t] = V

        pn = (self._K + 1) * self._V
        self._Poff = np.concatenate([[0], np.cumsum(pn)]).astype(np.int64)
        vn = self._V
        self._Voff = np.concatenate([[0], np.cumsum(vn)]).astype(np.int64)

        self._Pbuf = np.empty(int(self._Poff[-1]), dtype=np.float32)
        self._nbuf = np.empty(int(self._Voff[-1]), dtype=np.float64)
        self._ubuf = np.empty(int(self._Voff[-1]), dtype=np.float64)
        self._gbuf = np.empty(int(self._Voff[-1]), dtype=np.float64)
        self._fbuf = np.empty((int(self._Voff[-1]), self.M), dtype=np.float64)
        self._wbuf = np.empty(int(self._Voff[-1]), dtype=np.int64)

        for t in range(T):
            a, b = int(self._Poff[t]), int(self._Poff[t + 1])
            P = _normalise_rows(np.asarray(P_list[t], dtype=np.float64))
            self._Pbuf[a:b] = P.ravel().astype(np.float32)
            a, b = int(self._Voff[t]), int(self._Voff[t + 1])
            self._nbuf[a:b] = np.asarray(n_list[t], dtype=np.float64)
            uu = np.clip(np.asarray(u_list[t], dtype=np.float64), PROB_FLOOR, None)
            self._ubuf[a:b] = uu / uu.sum()
            self._gbuf[a:b] = np.asarray(g_list[t], dtype=np.float64)
            self._fbuf[a:b] = np.asarray(f_list[t], dtype=np.float64)
            self._wbuf[a:b] = (
                np.asarray(word_ids[t], dtype=np.int64)
                if word_ids is not None
                else np.arange(int(self._V[t]), dtype=np.int64)
            )

        if not np.all(np.isfinite(self._fbuf)):
            raise ValueError("non-finite value in the lexical feature matrix")
        if np.any(self._nbuf < 0):
            raise ValueError("negative response count in the corpus")

        self.target_slots = (
            np.asarray(target_slots, dtype=np.int64)
            if target_slots is not None
            else np.full(T, -1, dtype=np.int64)
        )
        if self.target_slots.shape != (T,):
            raise ValueError(
                f"target_slots has shape {self.target_slots.shape}, expected {(T,)}"
            )

        self._cache: dict[int, Target] = {}

    # -- access ---------------------------------------------------------

    def __len__(self) -> int:
        return self.n_targets

    def target(self, t: int) -> Target:
        """Materialise target ``t`` in float64, memoised."""
        hit = self._cache.get(t)
        if hit is not None:
            return hit
        K, V = int(self._K[t]), int(self._V[t])
        a, b = int(self._Poff[t]), int(self._Poff[t + 1])
        P = self._Pbuf[a:b].astype(np.float64).reshape(K + 1, V)
        P = _normalise_rows(P)
        a, b = int(self._Voff[t]), int(self._Voff[t + 1])
        tgt = Target(
            index=t,
            cluster=int(self.cluster_index[t]),
            P=P,
            n=self._nbuf[a:b].copy(),
            u=self._ubuf[a:b].copy(),
            f=self._fbuf[a:b].copy(),
            g=self._gbuf[a:b].copy(),
            D=np.diff(P, axis=0),
            word_ids=self._wbuf[a:b].copy(),
            target_slot=int(self.target_slots[t]),
        )
        self._cache[t] = tgt
        return tgt

    def __iter__(self) -> Iterator[Target]:
        for t in range(self.n_targets):
            yield self.target(t)

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- derived views --------------------------------------------------

    @property
    def total_responses(self) -> float:
        return float(self._nbuf.sum())

    @property
    def mean_K(self) -> float:
        return float(self._K.mean())

    def with_counts(self, counts: Sequence[np.ndarray]) -> "Corpus":
        """Return a copy carrying different response counts, same cache.

        This is how the ten readers share byte-identical code: only ``n``
        changes, and every synthetic reader gets exactly the response count its
        context has in the human data.
        """
        if len(counts) != self.n_targets:
            raise ValueError(
                f"expected {self.n_targets} count vectors, got {len(counts)}"
            )
        new = object.__new__(Corpus)
        new.__dict__.update(self.__dict__)
        new._nbuf = np.empty_like(self._nbuf)
        for t in range(self.n_targets):
            a, b = int(self._Voff[t]), int(self._Voff[t + 1])
            c = np.asarray(counts[t], dtype=np.float64)
            if c.shape != (int(self._V[t]),):
                raise ValueError(
                    f"target {t}: counts have shape {c.shape}, "
                    f"expected {(int(self._V[t]),)}"
                )
            new._nbuf[a:b] = c
        new._cache = {}
        return new

    def subset_clusters(self, cluster_index: Sequence[int]) -> "Corpus":
        """Resample clusters *with replacement*, for the paired cluster bootstrap.

        Targets are concatenated in the order the clusters are given, and each
        drawn copy receives a fresh cluster label, so a passage drawn twice
        contributes two independent clusters as the bootstrap requires.
        """
        P, n, u, f, g, cl, wid, ts = [], [], [], [], [], [], [], []
        for new_c, c in enumerate(cluster_index):
            members = np.flatnonzero(self.cluster_index == int(c))
            if members.size == 0:
                raise ValueError(f"cluster {c} has no targets")
            for t in members:
                tg = self.target(int(t))
                P.append(tg.P)
                n.append(tg.n)
                u.append(tg.u)
                f.append(tg.f)
                g.append(tg.g)
                cl.append(new_c)
                wid.append(tg.word_ids)
                ts.append(tg.target_slot)
        return Corpus(P, n, u, f, g, cl, wid, self.feature_names, ts)
