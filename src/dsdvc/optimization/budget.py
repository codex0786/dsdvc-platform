"""Marketing budget optimiser (spec §42).

    max E[Profit]  s.t.  sum_i Spend_i <= Budget,  lo_i <= Spend_i <= hi_i

Two solvers:
  1. response-curve optimisation: per-channel concave curves
         IncrementalConversions_i(s) = a_i (1 - exp(-s / b_i))
     fitted to simulator (or MMM / experiment) data, then solved with SLSQP; parameter
     uncertainty propagated by bootstrap -> interval on the optimum.
  2. simulation-based Bayesian optimisation over the allocation simplex using a
     Gaussian-process surrogate + expected improvement directly on simulated profit.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

from dsdvc.diffusion.abm import SimConfig
from dsdvc.experiments.design import run_batch

CHANNEL_MAP = {
    # channel -> function(config, total_spend) applying it to the simulator
    "instagram_paid": lambda c, s: c.paid_daily_budget.__setitem__("instagram", s / c.steps),
    "facebook_paid": lambda c, s: c.paid_daily_budget.__setitem__("facebook", s / c.steps),
    "creators": lambda c, s: setattr(c, "seeding_budget", s),
    "referral_whatsapp": lambda c, s: setattr(c, "referral_budget", s),
    "retention": lambda c, s: setattr(c, "retention_budget", s),
    "organic_content": lambda c, s: setattr(c, "content_budget", s),
}

CHANNEL_GET = {
    "instagram_paid": lambda c: c.paid_daily_budget.get("instagram", 0.0) * c.steps,
    "facebook_paid": lambda c: c.paid_daily_budget.get("facebook", 0.0) * c.steps,
    "creators": lambda c: c.seeding_budget,
    "referral_whatsapp": lambda c: c.referral_budget,
    "retention": lambda c: c.retention_budget or 0.0,
    "organic_content": lambda c: c.content_budget,
}


def current_allocation(cfg: SimConfig) -> dict[str, float]:
    return {k: float(g(cfg)) for k, g in CHANNEL_GET.items()}


def apply_allocation(cfg: SimConfig, alloc: dict[str, float], reset: bool = True) -> SimConfig:
    """reset=True zeroes every lever first (full allocation); reset=False changes only the given channels."""
    c = copy.deepcopy(cfg)
    if reset:
        c.paid_daily_budget = {"instagram": 0.0, "facebook": 0.0}
        c.seeding_budget = 0.0
        c.retention_budget = 0.0
        c.referral_budget = 0.0
        c.content_budget = 0.0
    for k, v in alloc.items():
        CHANNEL_MAP[k](c, float(v))
    return c


def _profit(cfg: SimConfig, alloc: dict, seeds, workers=1, reset: bool = True) -> pd.DataFrame:
    c = apply_allocation(cfg, alloc, reset)
    return run_batch(c, [{}], list(seeds), workers)


@dataclass
class ResponseCurve:
    channel: str
    a: float
    b: float
    a_se: float = 0.0
    b_se: float = 0.0

    def __call__(self, s):
        return self.a * (1 - np.exp(-np.asarray(s, dtype=float) / max(self.b, 1e-9)))


def fit_response_curve(channel: str, spend: np.ndarray, incremental: np.ndarray) -> ResponseCurve:
    spend, inc = np.asarray(spend, float), np.asarray(incremental, float)
    f = lambda s, a, b: a * (1 - np.exp(-s / b))  # noqa: E731
    try:
        (a, b), cov = optimize.curve_fit(f, spend, inc, p0=[max(inc.max(), 1) * 1.5, max(spend.mean(), 1)],
                                         bounds=([0, 1], [np.inf, np.inf]), maxfev=20000)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except (RuntimeError, ValueError):
        a, b, se = max(inc.max(), 0.0), max(spend.max(), 1.0), np.array([0.0, 0.0])
    return ResponseCurve(channel, float(a), float(b), float(se[0]), float(se[1]))


def curves_from_simulator(cfg: SimConfig, channels: list[str], levels: list[float], seeds=(0, 1, 2),
                          metric: str = "gross_margin", workers: int = 1,
                          baseline: dict[str, float] | None = None) -> tuple[dict[str, ResponseCurve], pd.DataFrame]:
    """Spend sweeps for each channel *on top of a baseline plan* (other channels held at the baseline,
    common random numbers). Holding others at zero would miss interactions, e.g. referral needs
    customers acquired by other channels. Default metric is gross margin so retention value counts."""
    baseline = baseline or {}
    base_cfg = apply_allocation(cfg, baseline, reset=False)
    base = _profit(base_cfg, {}, seeds, workers, reset=False)[metric].to_numpy()
    rows, curves = [], {}
    for ch in channels:
        sp, inc = [0.0], [0.0]
        for lv in levels:
            r = _profit(base_cfg, {ch: CHANNEL_GET[ch](base_cfg) + lv}, seeds, workers, reset=False)[metric].to_numpy()
            d = r - base
            rows.append({"channel": ch, "spend": lv, "incremental_mean": d.mean(), "incremental_sd": d.std()})
            sp += [lv] * len(d)
            inc += list(d)
        curves[ch] = fit_response_curve(ch, np.array(sp), np.array(inc))
    return curves, pd.DataFrame(rows)


def optimize_allocation(curves: dict[str, ResponseCurve], budget: float, value_per_conversion: float = 1.0,
                        bounds: dict[str, tuple[float, float]] | None = None, n_boot: int = 200,
                        seed: int = 0) -> dict:
    chans = list(curves)
    bounds = bounds or {}
    bnds = [bounds.get(c, (0.0, budget)) for c in chans]

    def solve(cs):
        def neg(x):
            return -(sum(cs[c](x[i]) for i, c in enumerate(chans)) * value_per_conversion - x.sum())
        x0 = np.full(len(chans), budget / len(chans))
        cons = [{"type": "ineq", "fun": lambda x: budget - x.sum()}]
        r = optimize.minimize(neg, x0, bounds=bnds, constraints=cons, method="SLSQP")
        return r.x, -r.fun

    x, val = solve(curves)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        cs = {c: ResponseCurve(c, max(0, rng.normal(k.a, k.a_se)), max(1, rng.normal(k.b, k.b_se))) for c, k in curves.items()}
        xb, _ = solve(cs)
        boots.append(xb)
    B = np.array(boots)
    alloc = pd.DataFrame({"channel": chans, "spend": x, "spend_low": np.quantile(B, 0.05, axis=0),
                          "spend_high": np.quantile(B, 0.95, axis=0),
                          "expected_incremental_value": [curves[c](x[i]) * value_per_conversion for i, c in enumerate(chans)],
                          "marginal_roi_at_optimum": [curves[c].a / curves[c].b * np.exp(-x[i] / curves[c].b)
                                                      * value_per_conversion - 1 for i, c in enumerate(chans)]})
    return {"allocation": alloc, "expected_profit": float(val), "total_spend": float(x.sum()), "budget": budget,
            "status": "RECOMMENDED",
            "note": "Optimal spend may be below budget when marginal ROI turns negative. Intervals reflect response-curve uncertainty."}


def bayesian_optimize(cfg: SimConfig, channels: list[str], budget: float, n_init: int = 8, n_iter: int = 12,
                      seeds=(0, 1), metric: str = "profit", seed: int = 0, workers: int = 1) -> dict:
    """GP + expected improvement over allocations x in the simplex (shares of budget, sum <= 1).
    The total spend share is a free dimension, so under-spending is allowed."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    rng = np.random.default_rng(seed)
    k = len(channels)

    def to_alloc(z):  # z in [0,1]^{k+1}: k mixing weights + total share
        w = np.asarray(z[:k]) + 1e-9
        w = w / w.sum()
        return {c: float(w[i] * z[k] * budget) for i, c in enumerate(channels)}

    def evaluate(z):
        df = _profit(cfg, to_alloc(z), seeds, workers)
        return float(df[metric].mean())

    Z = rng.random((n_init, k + 1))
    y = np.array([evaluate(z) for z in Z])
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(k + 1) * 0.5, nu=2.5) + WhiteKernel(1e-2)
    for _ in range(n_iter):
        mu_y, sd_y = y.mean(), y.std() + 1e-9
        gp = GaussianProcessRegressor(kernel, normalize_y=False, n_restarts_optimizer=2, random_state=seed)
        gp.fit(Z, (y - mu_y) / sd_y)
        cand = rng.random((2000, k + 1))
        m, s = gp.predict(cand, return_std=True)
        best = ((y - mu_y) / sd_y).max()
        imp = m - best - 0.01
        zsc = imp / np.maximum(s, 1e-9)
        ei = imp * stats.norm.cdf(zsc) + s * stats.norm.pdf(zsc)
        zn = cand[int(np.argmax(ei))]
        Z = np.vstack([Z, zn])
        y = np.append(y, evaluate(zn))
    i = int(np.argmax(y))
    trace = pd.DataFrame([{**to_alloc(z), metric: v} for z, v in zip(Z, y)])
    return {"best_allocation": to_alloc(Z[i]), f"best_{metric}": float(y[i]), "trace": trace,
            "evaluations": len(y), "status": "RECOMMENDED",
            "note": "Simulation-based Bayesian optimisation; profit is averaged over common random seeds."}
