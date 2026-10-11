"""Simple vs complex contagion and threshold cascades (spec §6, §8).

Simple contagion: each exposure is an independent chance to adopt,
    P = 1 - prod_k (1 - beta * w_k).
Complex contagion (Centola & Macy 2007): adoption requires social confirmation from
multiple *independent* sources; probability is a steep function of the number of
distinct adopting contacts relative to an individual threshold:
    P = sigmoid(kappa * (Signal_i - Threshold_i)),
    Signal_i = f(n_independent, TieStrength, Trust, SocialProof).

Whether a product behaves as a simple or complex contagion is an *empirical* question:
``fit_threshold_mixture`` estimates the share of complex adopters from data on the
number of adopting contacts at adoption time.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy import optimize

from dsdvc.core.literature import cites


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def simple_contagion_prob(weights: np.ndarray, beta: float) -> float:
    w = np.asarray(weights, dtype=float)
    return float(1 - np.prod(1 - np.clip(beta * w, 0, 1)))


@cites("centola_macy2007", "granovetter1978")
def complex_contagion_signal(n_independent: np.ndarray, tie_strength: np.ndarray, trust: np.ndarray,
                             social_proof: np.ndarray, w=(1.0, 0.5, 0.5, 0.8)) -> np.ndarray:
    """Effective confirmation signal in 'independent exposure' units."""
    a, b, c, d = w
    return (a * np.asarray(n_independent, dtype=float) * (1 + b * np.asarray(tie_strength))
            * (1 + c * (np.asarray(trust) - 0.5)) + d * np.asarray(social_proof))


@cites("centola_macy2007", "granovetter1978")
def complex_contagion_prob(signal: np.ndarray, threshold: np.ndarray, kappa: float = 4.0) -> np.ndarray:
    return sigmoid(kappa * (np.asarray(signal) - np.asarray(threshold)))


@dataclass
class CascadeResult:
    adopted: np.ndarray
    rounds: int
    size_by_round: list[int]

    @property
    def size(self) -> int:
        return int(self.adopted.sum())


@cites("kempe2003")
def independent_cascade(g: nx.Graph, seeds, p_edge: float | np.ndarray, seed: int = 0, max_rounds: int = 100) -> CascadeResult:
    rng = np.random.default_rng(seed)
    n = g.number_of_nodes()
    adopted = np.zeros(n, dtype=bool)
    frontier = list(seeds)
    adopted[frontier] = True
    sizes = [len(frontier)]
    r = 0
    while frontier and r < max_rounds:
        r += 1
        new = []
        for u in frontier:
            nbrs = g.successors(u) if g.is_directed() else g.neighbors(u)
            for v in nbrs:
                if not adopted[v]:
                    pe = p_edge if np.isscalar(p_edge) else p_edge[v]
                    if rng.random() < pe * g[u][v].get("weight", 1.0):
                        adopted[v] = True
                        new.append(v)
        frontier = new
        sizes.append(int(adopted.sum()))
    return CascadeResult(adopted, r, sizes)


@cites("granovetter1978", "watts_dodds2007", "centola_macy2007")
def threshold_cascade(g: nx.Graph, seeds, thresholds: np.ndarray, fractional: bool = True,
                      max_rounds: int = 200) -> CascadeResult:
    """Deterministic threshold cascade. ``fractional`` -> Watts-Dodds style fraction of
    neighbours; otherwise absolute count of adopting neighbours (Centola-Macy style)."""
    ug = g.to_undirected() if g.is_directed() else g
    n = ug.number_of_nodes()
    adopted = np.zeros(n, dtype=bool)
    adopted[list(seeds)] = True
    A = nx.to_scipy_sparse_array(ug, nodelist=range(n), weight=None, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    sizes = [int(adopted.sum())]
    for r in range(1, max_rounds + 1):
        k = A @ adopted.astype(float)
        signal = np.where(deg > 0, k / np.maximum(deg, 1), 0) if fractional else k
        new = (~adopted) & (signal >= thresholds) & (k > 0)
        if not new.any():
            return CascadeResult(adopted, r, sizes)
        adopted |= new
        sizes.append(int(adopted.sum()))
    return CascadeResult(adopted, max_rounds, sizes)


def fit_threshold_mixture(k_at_adoption: np.ndarray, max_k: int = 10) -> dict:
    """Estimate share of simple vs complex adopters from the number of adopting contacts
    each adopter had when adopting (observed or simulated data).

    Model: simple adopters' k ~ Poisson(lam1) truncated at >=0; complex adopters need k>=2
    and k-2 ~ Poisson(lam2). Fitted by maximum likelihood; returns pi_complex with a
    likelihood-ratio test against the pure-simple model.
    """
    from scipy import stats
    k = np.clip(np.asarray(k_at_adoption, dtype=int), 0, max_k)

    def nll(theta):
        pi = sigmoid(theta[0])
        l1, l2 = np.exp(theta[1]), np.exp(theta[2])
        f_simple = stats.poisson.pmf(k, l1)
        f_complex = np.where(k >= 2, stats.poisson.pmf(k - 2, l2), 0.0)
        return -np.log((1 - pi) * f_simple + pi * f_complex + 1e-300).sum()

    r = optimize.minimize(nll, x0=[0.0, 0.0, 0.0], method="Nelder-Mead", options={"maxiter": 4000})
    lam0 = max(k.mean(), 1e-6)
    from scipy import stats as st
    nll0 = -st.poisson.logpmf(k, lam0).sum()
    lr = 2 * (nll0 - r.fun)
    p = float(1 - st.chi2.cdf(max(lr, 0), df=2))
    return {"pi_complex": float(sigmoid(r.x[0])), "lambda_simple": float(np.exp(r.x[1])),
            "lambda_complex": float(np.exp(r.x[2])), "lr_stat": float(lr), "p_value_vs_simple": p}


@cites("centola_macy2007")
def long_ties_experiment(n: int = 400, k: int = 4, rewire_probs=(0.0, 0.05, 0.2, 0.6), threshold: int = 2,
                         reps: int = 5, seed: int = 0) -> list[dict]:
    """Reproduces the *qualitative* Centola-Macy result on ring lattices: random rewiring helps
    simple contagion but can hurt complex contagion (threshold >= 2 adopting neighbours)."""
    out = []
    for p in rewire_probs:
        cs, ss = [], []
        for r in range(reps):
            g = nx.watts_strogatz_graph(n, k, p, seed=seed + r)
            seeds = [0, 1, 2, 3]  # a clustered seed neighbourhood
            cs.append(threshold_cascade(g, seeds, np.full(n, threshold), fractional=False).size / n)
            ss.append(threshold_cascade(g, seeds, np.full(n, 1), fractional=False).size / n)
        out.append({"rewire_p": p, "complex_reach": float(np.mean(cs)), "simple_reach": float(np.mean(ss))})
    return out
