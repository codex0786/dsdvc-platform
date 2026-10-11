"""Seeding strategies and creator cost model (spec §8).

Strategies compared in simulation (Watts & Dodds 2007 hypothesis test):
  A celebrity_first   - highest-reach Instagram accounts
  B micro_network     - many mid-tier accounts (70th-97th percentile reach) with high relevance
  C community_seeding - within-community central members on Facebook/WhatsApp (samples/ambassadors)
  D hybrid            - budget split across A, B, C
  random              - control

Costs are ESTIMATED from a configurable rate card (``CostModel``), never observed.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from dsdvc.core.literature import cites
from dsdvc.networks.multiplex import MultiplexNetwork

STRATEGIES = ("celebrity_first", "micro_network", "community_seeding", "hybrid", "random")


@dataclass
class CostModel:
    """Rate-card prior (currency units). Defaults are illustrative placeholders - replace with
    quotes/benchmarks for the market (labelled ESTIMATED)."""
    base_fee: float = 2000.0
    per_1k_followers: float = 250.0
    celebrity_premium: float = 1.5      # multiplier above celebrity_threshold
    celebrity_threshold: float = 1_000_000
    community_seed_cost: float = 1500.0  # product samples + ambassador incentive
    population_scale: float = 100.0      # real people represented by one agent

    def creator_fee(self, in_degree: float) -> float:
        followers = in_degree * self.population_scale
        fee = self.base_fee + self.per_1k_followers * followers / 1000
        return fee * (self.celebrity_premium if followers >= self.celebrity_threshold else 1.0)


def _within_community_centrality(g: nx.Graph, community: np.ndarray) -> np.ndarray:
    ug = g.to_undirected() if g.is_directed() else g
    out = np.zeros(ug.number_of_nodes())
    for i in ug.nodes:
        out[i] = sum(1 for j in ug.neighbors(i) if community[j] == community[i])
    return out


@cites("watts_dodds2007", "deveirman2017", "kempe2003")
def select_seeds(mx: MultiplexNetwork, strategy: str, budget: float, costs: CostModel | None = None,
                 relevance: np.ndarray | None = None, seed: int = 0) -> dict:
    """Return {'seeds': [...], 'roles': {agent: role}, 'cost': total, 'fees': {agent: fee}}."""
    costs = costs or CostModel()
    rng = np.random.default_rng(seed)
    ig = mx.layers["instagram"]
    indeg = np.array([ig.in_degree(i) for i in range(mx.n)], dtype=float)
    rel = relevance if relevance is not None else np.ones(mx.n)
    chosen, roles, fees = [], {}, {}
    spent = 0.0

    def take(cands, role, unit_cost=None, limit=None):
        nonlocal spent
        k = 0
        for c in cands:
            c = int(c)
            if c in roles:
                continue
            fee = unit_cost if unit_cost is not None else costs.creator_fee(indeg[c])
            if spent + fee > budget + 1e-9:
                continue
            chosen.append(c)
            roles[c] = role
            fees[c] = fee
            spent += fee
            k += 1
            if limit and k >= limit:
                break

    if strategy == "celebrity_first":
        take(np.argsort(-indeg), "celebrity")
    elif strategy == "micro_network":
        lo, hi = np.quantile(indeg, [0.70, 0.97])
        cand = np.where((indeg >= lo) & (indeg <= hi))[0]
        take(cand[np.argsort(-(rel[cand] * np.log1p(indeg[cand])))], "micro")
    elif strategy == "community_seeding":
        cent = _within_community_centrality(mx.layers["whatsapp"], mx.community) + \
            _within_community_centrality(mx.layers["facebook"], mx.community)
        # round-robin across communities so every community gets seeded
        per_comm = {c: list(np.where(mx.community == c)[0][np.argsort(-cent[mx.community == c])])
                    for c in np.unique(mx.community)}
        while spent + costs.community_seed_cost <= budget and any(per_comm.values()):
            for c in list(per_comm):
                if per_comm[c]:
                    take([per_comm[c].pop(0)], "community", unit_cost=costs.community_seed_cost, limit=1)
    elif strategy == "hybrid":
        b0 = budget
        for strat, frac in (("celebrity_first", 0.4), ("micro_network", 0.4), ("community_seeding", 0.2)):
            sub = select_seeds(mx, strat, b0 * frac, costs, rel, seed)
            for s in sub["seeds"]:
                if s not in roles:
                    chosen.append(s)
                    roles[s] = sub["roles"][s]
                    fees[s] = sub["fees"][s]
                    spent += sub["fees"][s]
    elif strategy == "random":
        take(rng.permutation(mx.n), "random")
    else:
        raise ValueError(f"unknown strategy {strategy}")
    return {"seeds": chosen, "roles": roles, "cost": spent, "fees": fees, "strategy": strategy,
            "cost_status": "estimated (rate-card prior)"}
