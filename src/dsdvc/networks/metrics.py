"""Network structure metrics (spec §4, §16): homophily, modularity, clustering, assortativity,
conductance, density, cross-community exposure and bridge centrality."""
from __future__ import annotations

import networkx as nx
import numpy as np

from dsdvc.core.literature import cites


def edge_homophily(g: nx.Graph, labels: np.ndarray) -> float:
    edges = list(g.edges())
    if not edges:
        return 0.0
    return float(np.mean([labels[u] == labels[v] for u, v in edges]))


def coleman_homophily(g: nx.Graph, labels: np.ndarray) -> float:
    """Edge homophily normalised against the share expected under random mixing (0 = random)."""
    h = edge_homophily(g, labels)
    _, counts = np.unique(labels, return_counts=True)
    expected = float(((counts / counts.sum()) ** 2).sum())
    return float((h - expected) / (1 - expected)) if expected < 1 else 0.0


def partition_from_labels(labels: np.ndarray) -> list[set[int]]:
    return [set(np.where(labels == c)[0].tolist()) for c in np.unique(labels)]


@cites("newman2006")
def modularity(g: nx.Graph, labels: np.ndarray) -> float:
    ug = g.to_undirected() if g.is_directed() else g
    if ug.number_of_edges() == 0:
        return 0.0
    return float(nx.community.modularity(ug, partition_from_labels(labels)))


def assortativity(g: nx.Graph, labels: np.ndarray) -> float:
    ug = g.to_undirected() if g.is_directed() else g
    nx.set_node_attributes(ug, {i: int(labels[i]) for i in ug.nodes}, "lbl")
    try:
        v = nx.attribute_assortativity_coefficient(ug, "lbl")
    except (ZeroDivisionError, ValueError):
        return 0.0
    return float(v) if np.isfinite(v) else 0.0


def conductance_by_community(g: nx.Graph, labels: np.ndarray) -> dict[int, float]:
    ug = g.to_undirected() if g.is_directed() else g
    out = {}
    for c in np.unique(labels):
        S = set(np.where(labels == c)[0].tolist())
        if 0 < len(S) < ug.number_of_nodes():
            try:
                out[int(c)] = float(nx.conductance(ug, S))
            except ZeroDivisionError:
                out[int(c)] = float("nan")
    return out


def density_by_community(g: nx.Graph, labels: np.ndarray) -> dict[int, float]:
    ug = g.to_undirected() if g.is_directed() else g
    return {int(c): float(nx.density(ug.subgraph(np.where(labels == c)[0].tolist())))
            for c in np.unique(labels)}


def cross_community_exposure(g: nx.Graph, labels: np.ndarray) -> np.ndarray:
    """Per node: share of neighbours outside the node's community."""
    ug = g.to_undirected() if g.is_directed() else g
    out = np.zeros(ug.number_of_nodes())
    for i in ug.nodes:
        nb = list(ug.neighbors(i))
        if nb:
            out[i] = float(np.mean([labels[j] != labels[i] for j in nb]))
    return out


def bridge_centrality(g: nx.Graph, k_sample: int | None = 200, seed: int = 0) -> np.ndarray:
    """Bridging centrality (Hwang et al. 2006): betweenness x bridging coefficient.

    Bridging coefficient BC(v) = (1/deg v) / sum_{u in N(v)} 1/deg u. Betweenness is
    sample-approximated for large graphs (``k_sample``)."""
    ug = g.to_undirected() if g.is_directed() else g
    n = ug.number_of_nodes()
    k = None if (k_sample is None or n <= k_sample) else k_sample
    btw = nx.betweenness_centrality(ug, k=k, seed=seed, normalized=True)
    deg = dict(ug.degree())
    out = np.zeros(n)
    for v in ug.nodes:
        if deg[v] == 0:
            continue
        denom = sum(1.0 / deg[u] for u in ug.neighbors(v) if deg[u] > 0)
        bc = (1.0 / deg[v]) / denom if denom > 0 else 0.0
        out[v] = btw[v] * bc
    return out


def structure_report(g: nx.Graph, labels: np.ndarray, clustering_sample: int = 500, seed: int = 0) -> dict:
    ug = g.to_undirected() if g.is_directed() else g
    nodes = list(ug.nodes)
    rng = np.random.default_rng(seed)
    sample = nodes if len(nodes) <= clustering_sample else rng.choice(nodes, clustering_sample, replace=False).tolist()
    comms, sizes = np.unique(labels, return_counts=True)
    return {
        "nodes": ug.number_of_nodes(),
        "edges": ug.number_of_edges(),
        "density": float(nx.density(ug)),
        "avg_clustering": float(nx.average_clustering(ug, nodes=sample)),
        "edge_homophily": edge_homophily(ug, labels),
        "coleman_homophily": coleman_homophily(ug, labels),
        "modularity": modularity(ug, labels),
        "assortativity": assortativity(ug, labels),
        "community_count": int(len(comms)),
        "mean_community_size": float(sizes.mean()),
        "max_community_size": int(sizes.max()),
        "mean_cross_community_exposure": float(cross_community_exposure(ug, labels).mean()),
    }
