"""Community discovery (spec §16): Louvain, Leiden, label propagation, spectral, HDBSCAN.

Every result reports the algorithm actually used; if an optional backend is missing we
say so instead of silently substituting.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from dsdvc.core.literature import cites
from dsdvc.networks import metrics as M

METHODS = ("louvain", "leiden", "label_propagation", "spectral", "hdbscan")


@dataclass
class CommunityResult:
    method: str
    labels: np.ndarray
    modularity: float
    per_community: list[dict] = field(default_factory=list)
    bridges: list[int] = field(default_factory=list)
    isolated_clusters: list[int] = field(default_factory=list)
    micro_communities: list[int] = field(default_factory=list)
    notes: str = ""

    @property
    def n_communities(self) -> int:
        return int(len(np.unique(self.labels[self.labels >= 0])))


def _labels_from_sets(n: int, comms) -> np.ndarray:
    lab = np.full(n, -1, dtype=int)
    for c, s in enumerate(sorted(comms, key=len, reverse=True)):
        for v in s:
            lab[int(v)] = c
    return lab


def _spectral_embedding(ug: nx.Graph, dim: int, seed: int) -> np.ndarray:
    from sklearn.manifold import SpectralEmbedding
    A = nx.to_scipy_sparse_array(ug, nodelist=range(ug.number_of_nodes()), format="csr")
    A = (A + A.T).tocsr()
    A.data[:] = 1.0
    A.indices = A.indices.astype(np.int32)
    A.indptr = A.indptr.astype(np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # disconnected graphs are expected
        return SpectralEmbedding(n_components=dim, affinity="precomputed", random_state=seed).fit_transform(A)


@cites("blondel2008", "traag2019", "newman2006")
def detect_communities(g: nx.Graph, method: str = "louvain", seed: int = 0, resolution: float = 1.0,
                       k: int | None = None, bridge_top: int = 20, micro_size: int = 5) -> CommunityResult:
    ug = g.to_undirected() if g.is_directed() else g
    n = ug.number_of_nodes()
    notes = ""
    if method == "louvain":
        labels = _labels_from_sets(n, nx.community.louvain_communities(ug, seed=seed, resolution=resolution))
    elif method == "leiden":
        try:
            import igraph as ig
            import leidenalg
            G = ig.Graph(n=n, edges=list(ug.edges()))
            part = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition,
                                            resolution_parameter=resolution, seed=seed)
            labels = np.array(part.membership, dtype=int)
            notes = "leidenalg backend"
        except ImportError:
            labels = _labels_from_sets(n, nx.community.leiden_communities(ug, seed=seed, resolution=resolution))
            notes = "networkx leiden backend"
    elif method == "label_propagation":
        labels = _labels_from_sets(n, nx.community.label_propagation_communities(ug))
    elif method == "spectral":
        from sklearn.cluster import KMeans
        dim = min(16, max(2, n // 20))
        emb = _spectral_embedding(ug, dim, seed)
        if k is None:
            # choose k by modularity over a small range
            best = None
            for kk in range(2, min(16, n // 10) + 1):
                lab = KMeans(kk, n_init=4, random_state=seed).fit_predict(emb)
                q = M.modularity(ug, lab)
                if best is None or q > best[0]:
                    best = (q, lab, kk)
            labels, notes = best[1], f"k selected by modularity: {best[2]}"
        else:
            labels = KMeans(k, n_init=4, random_state=seed).fit_predict(emb)
    elif method == "hdbscan":
        from sklearn.cluster import HDBSCAN
        emb = _spectral_embedding(ug, min(8, max(2, n // 50)), seed)
        labels = HDBSCAN(min_cluster_size=max(5, n // 100), copy=True).fit_predict(emb)
        notes = "HDBSCAN on spectral embedding; label -1 = noise (unassigned)"
    else:
        raise ValueError(f"unknown method {method}; choose from {METHODS}")

    assigned = labels.copy()
    if (assigned < 0).any():  # treat noise as singleton group for modularity
        assigned[assigned < 0] = assigned.max() + 1 + np.arange((assigned < 0).sum())
    q = M.modularity(ug, assigned)
    cond = M.conductance_by_community(ug, assigned)
    dens = M.density_by_community(ug, assigned)
    per = []
    isolated, micro = [], []
    for c in np.unique(labels):
        if c < 0:
            continue
        size = int((labels == c).sum())
        per.append({"community": int(c), "size": size, "density": dens.get(int(c), 0.0),
                    "conductance": cond.get(int(c), float("nan"))})
        if size <= micro_size:
            micro.append(int(c))
        if cond.get(int(c), 1.0) < 0.05:
            isolated.append(int(c))
    bc = M.bridge_centrality(ug, seed=seed)
    bridges = np.argsort(-bc)[:bridge_top].tolist()
    return CommunityResult(method=method, labels=labels, modularity=q, per_community=per,
                           bridges=[int(b) for b in bridges], isolated_clusters=isolated,
                           micro_communities=micro, notes=notes)


def compare_methods(g: nx.Graph, seed: int = 0, methods=METHODS) -> list[dict]:
    rows = []
    for m in methods:
        r = detect_communities(g, m, seed=seed)
        rows.append({"method": m, "communities": r.n_communities, "modularity": r.modularity, "notes": r.notes})
    return rows


def community_dynamics(prev: np.ndarray, curr: np.ndarray, growth_threshold: float = 0.2) -> list[dict]:
    """Match communities across snapshots by Jaccard overlap; flag emerging / declining clusters."""
    out = []
    prev_sets = {c: set(np.where(prev == c)[0]) for c in np.unique(prev) if c >= 0}
    for c in np.unique(curr):
        if c < 0:
            continue
        s = set(np.where(curr == c)[0])
        best, jac = None, 0.0
        for pc, ps in prev_sets.items():
            j = len(s & ps) / max(1, len(s | ps))
            if j > jac:
                best, jac = pc, j
        if best is None or jac < 0.1:
            trend = "emerging"
            growth = float("inf")
        else:
            growth = (len(s) - len(prev_sets[best])) / max(1, len(prev_sets[best]))
            trend = "growing" if growth > growth_threshold else ("declining" if growth < -growth_threshold else "stable")
        out.append({"community": int(c), "matched_previous": best, "jaccard": jac, "size": len(s),
                    "growth": growth, "trend": trend})
    return out
