"""Competitive market: multinomial logit with network effects (spec §26, §27).

    U_ij = b1 Quality_j - b2 Price_j + b3 Brand_j + b4 SocialProof_j + NetworkUtility_j
           + b6 CommunityFit_ij + b7 Exposure_ij + e_ij
    P_ij = exp(U_ij) / sum_k exp(U_ik)         (McFadden 1974), optional outside option.

NetworkUtility_j specifications compared on data:
    log:   a log(1 + N_j)       (baseline)
    power: a N_j^g
    none:  0
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

from dsdvc.core.literature import cites

ATTRS = ["quality", "price", "brand", "social_proof", "community_fit", "exposure"]


def network_utility(N: np.ndarray, spec: str, alpha: float, gamma: float = 0.5) -> np.ndarray:
    N = np.asarray(N, dtype=float)
    if spec == "log":
        return alpha * np.log1p(N)
    if spec == "power":
        return alpha * np.power(np.maximum(N, 0), gamma)
    if spec == "none":
        return np.zeros_like(N)
    raise ValueError(spec)


@dataclass
class MNLParams:
    beta: dict[str, float] = field(default_factory=lambda: {"quality": 1.5, "price": -1.2, "brand": 1.0,
                                                            "social_proof": 0.8, "community_fit": 1.0, "exposure": 0.9})
    alpha: float = 0.15
    gamma: float = 0.5
    network_spec: str = "log"
    outside_option: float = 0.0  # utility of not buying (None to disable)


@cites("mcfadden1974", "katz_shapiro1985")
def choice_probabilities(X: np.ndarray, N: np.ndarray, params: MNLParams, include_outside: bool = True) -> np.ndarray:
    """X: (n_consumers, n_brands, n_attrs) in ATTRS order; N: (n_brands,) installed base (scaled).
    Returns (n_consumers, n_brands [+1 outside])."""
    b = np.array([params.beta[a] for a in ATTRS])
    NU = network_utility(N, params.network_spec, params.alpha, params.gamma)
    U = X @ b + (NU[None, :] if NU.ndim == 1 else NU)
    if include_outside:
        U = np.concatenate([U, np.full((U.shape[0], 1), params.outside_option)], axis=1)
    U -= U.max(axis=1, keepdims=True)
    E = np.exp(U)
    return E / E.sum(axis=1, keepdims=True)


@cites("mcfadden1974", "katz_shapiro1985")
def fit_mnl(X: np.ndarray, N: np.ndarray, choice: np.ndarray, spec: str = "log", include_outside: bool = True) -> dict:
    """Maximum likelihood for beta, alpha (and gamma for 'power'). ``choice`` indexes brands,
    with n_brands meaning the outside option. N may be (n_brands,) or (n_obs, n_brands)."""
    n_obs, n_brands, k = X.shape
    N2 = np.broadcast_to(np.asarray(N, float), (n_obs, n_brands))

    def unpack(theta):
        beta = theta[:k]
        alpha = theta[k] if spec != "none" else 0.0
        gamma = 1 / (1 + np.exp(-theta[k + 1])) if spec == "power" else 0.5
        return beta, alpha, gamma

    def nll(theta):
        beta, alpha, gamma = unpack(theta)
        U = X @ beta + network_utility(N2, spec, alpha, gamma)
        if include_outside:
            U = np.concatenate([U, np.zeros((n_obs, 1))], axis=1)
        lse = np.logaddexp.reduce(U, axis=1)
        return -(U[np.arange(n_obs), choice] - lse).sum() + 1e-4 * np.sum(theta ** 2)

    n_par = k + (0 if spec == "none" else 1) + (1 if spec == "power" else 0)
    r = optimize.minimize(nll, np.zeros(n_par), method="BFGS", options={"gtol": 1e-3})
    beta, alpha, gamma = unpack(r.x)
    se = np.sqrt(np.clip(np.diag(r.hess_inv), 0, None)) if hasattr(r, "hess_inv") else np.full(n_par, np.nan)
    ll = -float(r.fun)
    return {"spec": spec, "beta": dict(zip(ATTRS, beta.round(4).tolist())), "alpha": float(alpha),
            "gamma": float(gamma), "se": se.tolist(), "loglik": ll, "aic": 2 * n_par - 2 * ll,
            "bic": n_par * np.log(n_obs) - 2 * ll, "n_params": n_par, "converged": bool(r.success)}


def compare_network_specs(X, N, choice, include_outside: bool = True) -> pd.DataFrame:
    rows = [fit_mnl(X, N, choice, s, include_outside) for s in ("none", "log", "power")]
    df = pd.DataFrame([{k: r[k] for k in ("spec", "alpha", "gamma", "loglik", "aic", "bic", "converged")} for r in rows])
    return df.sort_values("aic").reset_index(drop=True)


def simulate_choices(X: np.ndarray, N: np.ndarray, params: MNLParams, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    P = choice_probabilities(X, N, params, include_outside=True)
    c = P.cumsum(axis=1)
    u = rng.random((P.shape[0], 1))
    return (u > c).sum(axis=1)


def market_shares(X: np.ndarray, N: np.ndarray, params: MNLParams, brands: list[str]) -> pd.Series:
    P = choice_probabilities(X, N, params, include_outside=True).mean(axis=0)
    return pd.Series(P, index=brands + ["no_purchase"])
