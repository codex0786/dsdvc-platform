"""Schelling-inspired endogenous homophily and community formation (spec §4).

This is NOT Schelling's residential model. We borrow its core insight - that mild
individual similarity preferences, acting through local moves, can produce strongly
separated aggregate structure - and apply it to *tie rewiring* in a social layer.

    Satisfaction_i = f(Similarity of neighbours)               (local similarity share)
    P(rewire_i)    = f(CommunityFit_i, Threshold_i, SwitchingCost_i)

Individual preference (``threshold``) is a separate input from the aggregate outcome
(homophily, modularity, ...), which is *measured* after the dynamics run.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from dsdvc.core.literature import cites
from dsdvc.networks import metrics as M


def neighbour_similarity(g: nx.Graph, taste: np.ndarray, i: int) -> float:
    nb = list(g.successors(i)) if g.is_directed() else list(g.neighbors(i))
    if not nb:
        return 0.5
    sims = taste[nb] @ taste[i]
    return float(np.clip((sims.mean() + 1) / 2, 0, 1))  # cosine mapped to [0, 1]


@dataclass
class SchellingParams:
    threshold_mean: float = 0.55   # individual preference for similar neighbours
    threshold_sd: float = 0.10
    switching_cost_mean: float = 0.4
    rewiring_rate: float = 0.05
    candidate_pool: int = 15        # how many candidate new ties an agent considers
    triadic_bias: float = 0.5       # share of candidates found via friends-of-friends


@cites("schelling1971", "mcpherson2001", "aral2009")
class SchellingRewirer:
    """Rewire ties of agents whose neighbourhood similarity falls below their threshold."""

    def __init__(self, g: nx.Graph, taste: np.ndarray, params: SchellingParams | None = None, seed: int = 0):
        self.g = g
        self.taste = taste
        self.p = params or SchellingParams()
        self.rng = np.random.default_rng(seed)
        n = g.number_of_nodes()
        self.threshold = np.clip(self.rng.normal(self.p.threshold_mean, self.p.threshold_sd, n), 0.05, 0.95)
        self.switching_cost = np.clip(self.rng.beta(2, 2, n) * 2 * self.p.switching_cost_mean, 0, 0.95)

    def fit(self) -> np.ndarray:
        return np.array([neighbour_similarity(self.g, self.taste, i) for i in range(self.g.number_of_nodes())])

    def step(self) -> int:
        g, rng = self.g, self.rng
        n = g.number_of_nodes()
        fit = self.fit()
        unhappy = np.where(fit < self.threshold)[0]
        p_move = self.p.rewiring_rate * (1 - self.switching_cost[unhappy])
        movers = unhappy[rng.random(len(unhappy)) < p_move]
        moves = 0
        for i in movers:
            i = int(i)
            nb = list(g.successors(i)) if g.is_directed() else list(g.neighbors(i))
            if not nb:
                continue
            sims = self.taste[nb] @ self.taste[i]
            drop = nb[int(np.argmin(sims))]
            cands = []
            fof = [w for v in nb for w in (g.successors(v) if g.is_directed() else g.neighbors(v))]
            k_tri = int(self.p.candidate_pool * self.p.triadic_bias)
            if fof:
                cands += rng.choice(fof, size=min(k_tri, len(fof)), replace=False).tolist()
            cands += rng.integers(0, n, self.p.candidate_pool - len(cands)).tolist()
            cands = [c for c in set(cands) if c != i and not g.has_edge(i, c)]
            if not cands:
                continue
            new = int(cands[int(np.argmax(self.taste[cands] @ self.taste[i]))])
            if self.taste[new] @ self.taste[i] <= self.taste[drop] @ self.taste[i]:
                continue
            w = g[i][drop].get("weight", 0.5)
            g.remove_edge(i, drop)
            g.add_edge(i, new, weight=w)
            moves += 1
        return moves

    def run(self, steps: int, labels: np.ndarray, record_every: int = 1) -> list[dict]:
        hist = []
        moves = 0
        for t in range(steps + 1):
            if t % record_every == 0:
                rep = {"step": t, "moves_since_last": moves, "mean_fit": float(self.fit().mean()),
                       "edge_homophily": M.edge_homophily(self.g, labels),
                       "modularity": M.modularity(self.g, labels),
                       "assortativity": M.assortativity(self.g, labels),
                       "mean_cross_community_exposure": float(M.cross_community_exposure(self.g, labels).mean())}
                hist.append(rep)
                moves = 0
            if t < steps:
                moves += self.step()
        return hist
