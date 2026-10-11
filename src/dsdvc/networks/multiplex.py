"""Multiplex social network G = {G_IG, G_FB, G_WA} (spec §9-§12).

The three layers are structurally different by construction:

* Instagram - directed follow graph with heavy-tailed in-degree (creators), weak ties.
  Diffusion is social (followers) *and* algorithmic (see ``recommendation.exposure``).
* Facebook - undirected friendship graph with high clustering plus explicit Groups
  (bipartite membership) for community-level diffusion.
* WhatsApp - sparse undirected strong-tie graph plus small group chats; high tie strength.

All synthetic layers are *priors* labelled SYNTHETIC. When authorised edge lists or
aggregate degree/clustering statistics are available, use :func:`from_edge_lists` or
calibrate the generator parameters (``calibration``). We never reconstruct private
graphs (spec §13).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from scipy import sparse

from dsdvc.core.literature import cites
from dsdvc.core.provenance import Provenance

LAYERS = ("instagram", "facebook", "whatsapp")


@dataclass
class LayerSpec:
    mean_degree: float
    homophily: float           # probability an edge is drawn within-community
    degree_heterogeneity: float  # lognormal sigma of activity weights
    tie_strength: tuple[float, float]  # beta(a, b) for edge weights
    directed: bool = False
    triadic_closure: float = 0.0
    n_groups: int = 0          # FB groups / WA group chats
    group_size: tuple[int, int] = (5, 30)


DEFAULT_SPECS: dict[str, LayerSpec] = {
    "instagram": LayerSpec(mean_degree=14, homophily=0.55, degree_heterogeneity=1.3,
                           tie_strength=(1.5, 5.0), directed=True, triadic_closure=0.05),
    "facebook": LayerSpec(mean_degree=10, homophily=0.70, degree_heterogeneity=0.6,
                          tie_strength=(2.5, 3.0), triadic_closure=0.35, n_groups=0, group_size=(10, 80)),
    "whatsapp": LayerSpec(mean_degree=5, homophily=0.80, degree_heterogeneity=0.35,
                          tie_strength=(5.0, 1.8), triadic_closure=0.5, n_groups=0, group_size=(3, 12)),
}


@dataclass
class MultiplexNetwork:
    n: int
    layers: dict[str, nx.Graph]
    community: np.ndarray                 # latent taste community (ground truth for synthetic)
    taste: np.ndarray                     # n x d preference vectors
    groups: dict[str, list[np.ndarray]] = field(default_factory=dict)  # FB groups / WA chats
    provenance: Provenance | None = None
    network_seed: int = 0

    def adjacency(self, layer: str, weighted: bool = True) -> sparse.csr_matrix:
        """A[i, j] > 0 means content from j can reach i (row = receiver)."""
        g = self.layers[layer]
        A = nx.to_scipy_sparse_array(g, nodelist=range(self.n), weight="weight" if weighted else None,
                                     format="csr")
        if g.is_directed():
            # edge j->i stored as follower relation i follows j: we store edge (follower, creator)
            # so row=follower, col=creator already = receiver x sender.
            return sparse.csr_matrix(A)
        return sparse.csr_matrix(A)

    def undirected(self, layer: str) -> nx.Graph:
        g = self.layers[layer]
        return g.to_undirected() if g.is_directed() else g

    def degree_stats(self) -> dict[str, dict[str, float]]:
        out = {}
        for name, g in self.layers.items():
            degs = np.array([d for _, d in (g.in_degree() if g.is_directed() else g.degree())])
            out[name] = {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
                         "mean_degree": float(degs.mean()), "max_degree": int(degs.max()),
                         "density": nx.density(g)}
        return out


def _activity(rng, n, sigma):
    w = rng.lognormal(0, sigma, n)
    return w / w.sum()


@cites("mcpherson2001", "goldenberg2001")
def _layer(rng, n, community, spec: LayerSpec) -> nx.Graph:
    g = nx.DiGraph() if spec.directed else nx.Graph()
    g.add_nodes_from(range(n))
    act = _activity(rng, n, spec.degree_heterogeneity)
    members = {c: np.where(community == c)[0] for c in np.unique(community)}
    member_w = {c: act[idx] / act[idx].sum() for c, idx in members.items()}
    m_target = int(spec.mean_degree * n / (1 if spec.directed else 2))
    a, b = spec.tie_strength
    tries = 0
    m = 0
    while m < m_target and tries < m_target * 6:
        tries += 1
        # receiver (follower) chosen uniformly-ish; sender/target by activity (popularity)
        i = int(rng.integers(n)) if spec.directed else int(rng.choice(n, p=act))
        nb = (list(g.successors(i)) if spec.directed else list(g.neighbors(i))) if spec.triadic_closure > 0 else []
        if nb and rng.random() < spec.triadic_closure:
            k = int(rng.choice(nb))
            nb2 = list(g.successors(k)) if spec.directed else list(g.neighbors(k))
            if not nb2:
                continue
            j = int(rng.choice(nb2))
        elif rng.random() < spec.homophily:
            c = community[i]
            j = int(rng.choice(members[c], p=member_w[c]))
        else:
            j = int(rng.choice(n, p=act))
        if i == j or g.has_edge(i, j):
            continue
        g.add_edge(i, j, weight=float(np.clip(rng.beta(a, b), 0.02, 1.0)))
        m += 1
    return g


def _groups(rng, n, community, n_groups, size_range, homophily) -> list[np.ndarray]:
    out = []
    members = {c: np.where(community == c)[0] for c in np.unique(community)}
    for _ in range(n_groups):
        size = int(rng.integers(size_range[0], size_range[1] + 1))
        c = int(rng.choice(list(members)))
        k_in = int(round(size * homophily))
        inside = rng.choice(members[c], size=min(k_in, len(members[c])), replace=False)
        outside = rng.choice(n, size=size - len(inside), replace=False)
        out.append(np.unique(np.concatenate([inside, outside])).astype(int))
    return out


@cites("mcpherson2001")
def synthetic_multiplex(n: int = 1000, n_communities: int = 8, homophily_scale: float = 1.0,
                        specs: dict[str, LayerSpec] | None = None, taste_dim: int = 4,
                        seed: int = 0) -> MultiplexNetwork:
    """Generate a SYNTHETIC multiplex network with planted communities.

    ``homophily_scale`` multiplies each layer's within-community probability (clipped to 0.98),
    letting experiments ask "what if the network becomes more segregated?" (§31).
    """
    rng = np.random.default_rng(seed)
    specs = specs or DEFAULT_SPECS
    sizes = rng.dirichlet(np.ones(n_communities) * 3) * n
    community = np.repeat(np.arange(n_communities), np.maximum(1, np.round(sizes).astype(int)))[:n]
    if len(community) < n:
        community = np.concatenate([community, rng.integers(0, n_communities, n - len(community))])
    rng.shuffle(community)
    centers = rng.normal(0, 1, (n_communities, taste_dim))
    taste = centers[community] + rng.normal(0, 0.6, (n, taste_dim))
    taste /= np.linalg.norm(taste, axis=1, keepdims=True) + 1e-9

    layers = {}
    groups = {}
    for name in LAYERS:
        s = specs[name]
        s2 = LayerSpec(**{**s.__dict__, "homophily": float(np.clip(s.homophily * homophily_scale, 0.0, 0.98))})
        layers[name] = _layer(rng, n, community, s2)
    groups["facebook"] = _groups(rng, n, community, max(4, n // 60), (10, 80), min(0.98, 0.7 * homophily_scale))
    groups["whatsapp"] = _groups(rng, n, community, max(6, n // 15), (3, 12), min(0.98, 0.85 * homophily_scale))
    return MultiplexNetwork(n=n, layers=layers, community=community, taste=taste, groups=groups,
                            provenance=Provenance.synthetic("synthetic_multiplex", seed), network_seed=seed)


def from_edge_lists(n: int, edges: dict[str, list[tuple[int, int, float]]], community: np.ndarray | None = None,
                    provenance: Provenance | None = None, directed_layers=("instagram",)) -> MultiplexNetwork:
    """Build a multiplex from authorised / user-provided edge lists (node ids 0..n-1)."""
    layers = {}
    for name in LAYERS:
        g = nx.DiGraph() if name in directed_layers else nx.Graph()
        g.add_nodes_from(range(n))
        for u, v, w in edges.get(name, []):
            g.add_edge(int(u), int(v), weight=float(w))
        layers[name] = g
    comm = community if community is not None else np.zeros(n, dtype=int)
    taste = np.zeros((n, 1))
    return MultiplexNetwork(n=n, layers=layers, community=comm, taste=taste, provenance=provenance)
