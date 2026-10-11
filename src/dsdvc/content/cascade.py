"""CascadeEngine (spec §22): reshare trees -> generations, depth, breadth, diversity,
structural virality, cross-community and cross-platform spread."""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from dsdvc.core.literature import cites


@cites("goel2016")
class CascadeEngine:
    """Build from share events with columns: share_id, parent_share_id (None for seed),
    sharer_id, platform, timestamp, optional community, optional target_platform."""

    def __init__(self, shares: pd.DataFrame):
        self.df = shares.copy()
        self.tree = nx.DiGraph()
        for r in self.df.itertuples(index=False):
            self.tree.add_node(r.share_id, sharer=r.sharer_id, platform=r.platform,
                               community=getattr(r, "community", None))
            if pd.notna(r.parent_share_id) and r.parent_share_id is not None:
                self.tree.add_edge(r.parent_share_id, r.share_id)
        self.roots = [n for n in self.tree.nodes if self.tree.in_degree(n) == 0]
        self.generation = {}
        for root in self.roots:
            for node, depth in nx.single_source_shortest_path_length(self.tree, root).items():
                self.generation[node] = depth

    def generations(self) -> dict[int, int]:
        counts = defaultdict(int)
        for g in self.generation.values():
            counts[g] += 1
        return dict(sorted(counts.items()))

    def reproduction_ratio(self) -> float:
        """SecondaryShares / PrimaryShares (generation >=2 over generation 1)."""
        g = self.generations()
        primary = g.get(1, 0)
        secondary = sum(v for k, v in g.items() if k >= 2)
        return float(secondary / primary) if primary else 0.0

    def depth(self) -> int:
        return max(self.generation.values(), default=0)

    def breadth(self) -> int:
        return max(self.generations().values(), default=0)

    def structural_virality(self, max_nodes: int = 2000, seed: int = 0) -> float:
        """Mean pairwise shortest-path distance in the (undirected) cascade tree (Goel et al. 2016).
        Sampled for large trees."""
        u = self.tree.to_undirected()
        vals = []
        rng = np.random.default_rng(seed)
        for comp in nx.connected_components(u):
            if len(comp) < 2:
                continue
            sub = u.subgraph(comp)
            nodes = list(comp)
            src = nodes if len(nodes) <= 200 else list(rng.choice(nodes, 200, replace=False))
            d = []
            for s in src[: max_nodes]:
                d += list(nx.single_source_shortest_path_length(sub, s).values())
            vals.append(np.sum(d) / max(1, len(src) * (len(nodes) - 1)))
        return float(np.mean(vals)) if vals else 0.0

    def diversity(self, attr: str = "community") -> float:
        """Shannon entropy (nats) of the attribute distribution among sharers."""
        vals = [d.get(attr) for _, d in self.tree.nodes(data=True) if d.get(attr) is not None]
        if not vals:
            return 0.0
        _, c = np.unique(vals, return_counts=True)
        p = c / c.sum()
        return float(-(p * np.log(p)).sum())

    def cross_edge_share(self, attr: str) -> float:
        edges = list(self.tree.edges())
        if not edges:
            return 0.0
        n = self.tree.nodes
        diff = [n[a].get(attr) != n[b].get(attr) for a, b in edges
                if n[a].get(attr) is not None and n[b].get(attr) is not None]
        return float(np.mean(diff)) if diff else 0.0

    def summary(self) -> dict:
        return {"size": self.tree.number_of_nodes(), "roots": len(self.roots), "depth": self.depth(),
                "breadth": self.breadth(), "generations": self.generations(),
                "reproduction_ratio": self.reproduction_ratio(),
                "structural_virality": self.structural_virality(),
                "community_diversity": self.diversity("community"),
                "cross_community_spread": self.cross_edge_share("community"),
                "cross_platform_spread": self.cross_edge_share("platform")}
