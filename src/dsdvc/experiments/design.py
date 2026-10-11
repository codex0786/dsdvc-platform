"""Experimental designs for simulation studies (spec §32).

A/B simulation, parameter sweeps, Monte Carlo, Latin hypercube, Sobol (Saltelli/Jansen
estimators) and Morris elementary effects. Each experiment returns a tidy frame plus the
run ids of every simulation so recommendations can be traced back (spec §39).
"""
from __future__ import annotations

import copy
import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

from dsdvc.core.literature import cites
from dsdvc.diffusion.abm import SimConfig, run_simulation

Setter = Callable[[SimConfig, float], None]


def _child(obj, key: str):
    if isinstance(obj, dict):
        return obj[key]
    if isinstance(obj, list):
        return obj[int(key)]
    return getattr(obj, key)


def set_param(cfg: SimConfig, name: str, value) -> None:
    """Set a (possibly nested) config parameter, e.g. 'platforms.instagram.algo_reach',
    'stepps.emotion', 'paid_daily_budget.instagram' or 'brands.1.price'."""
    parts = name.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = _child(obj, p)
    if isinstance(obj, dict):
        obj[parts[-1]] = value
    elif isinstance(obj, list):
        obj[int(parts[-1])] = value
    else:
        setattr(obj, parts[-1], value)


def get_param(cfg: SimConfig, name: str):
    obj = cfg
    for p in name.split("."):
        obj = _child(obj, p)
    return obj


def _run_one(args):
    cfg, overrides, seed = args
    c = copy.deepcopy(cfg)
    for k, v in overrides.items():
        set_param(c, k, v)
    c.seed = seed
    res = run_simulation(c)
    s = res.summary()
    return {**overrides, "seed": seed, **{k: v for k, v in s.items() if not isinstance(v, dict)},
            **{f"acq_{k}": v for k, v in s["acquisition_share_by_platform"].items()}}


def run_batch(cfg: SimConfig, designs: list[dict], seeds: list[int], workers: int = 1) -> pd.DataFrame:
    jobs = [(cfg, d, s) for d in designs for s in seeds]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_run_one, jobs))
    else:
        rows = [_run_one(j) for j in jobs]
    return pd.DataFrame(rows)


def monte_carlo(cfg: SimConfig, n_seeds: int = 20, overrides: dict | None = None, workers: int = 1) -> pd.DataFrame:
    """Stochastic replication: same parameters, different random seeds (network fixed)."""
    return run_batch(cfg, [overrides or {}], list(range(cfg.seed, cfg.seed + n_seeds)), workers)


def sweep(cfg: SimConfig, grid: dict[str, list], reps: int = 3, workers: int = 1) -> pd.DataFrame:
    keys = list(grid)
    designs = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]
    return run_batch(cfg, designs, list(range(cfg.seed, cfg.seed + reps)), workers)


def latin_hypercube(cfg: SimConfig, bounds: dict[str, tuple[float, float]], n: int = 32, reps: int = 1,
                    seed: int = 0, workers: int = 1) -> pd.DataFrame:
    s = qmc.LatinHypercube(d=len(bounds), seed=seed).random(n)
    lo = np.array([b[0] for b in bounds.values()])
    hi = np.array([b[1] for b in bounds.values()])
    X = qmc.scale(s, lo, hi)
    designs = [dict(zip(bounds, row)) for row in X]
    return run_batch(cfg, designs, list(range(cfg.seed, cfg.seed + reps)), workers)


@cites("saltelli2010")
def sobol_indices(f: Callable[[np.ndarray], np.ndarray], bounds: dict[str, tuple[float, float]], n: int = 256,
                  seed: int = 0, n_boot: int = 200) -> pd.DataFrame:
    """First-order (Saltelli 2010) and total (Jansen) Sobol indices with bootstrap CIs.
    ``f`` maps an (N, d) array of parameter values to N outputs. Cost: n*(d+2) evaluations."""
    d = len(bounds)
    lo = np.array([b[0] for b in bounds.values()])
    hi = np.array([b[1] for b in bounds.values()])
    base = qmc.Sobol(2 * d, scramble=True, seed=seed).random(n)
    A = qmc.scale(base[:, :d], lo, hi)
    B = qmc.scale(base[:, d:], lo, hi)
    fA, fB = f(A), f(B)
    fAB = []
    for i in range(d):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        fAB.append(f(ABi))
    fAB = np.array(fAB)

    def est(idx):
        a, b, ab = fA[idx], fB[idx], fAB[:, idx]
        var = np.var(np.concatenate([a, b]))
        if var == 0:
            return np.zeros(d), np.zeros(d)
        S1 = np.mean(b[None, :] * (ab - a[None, :]), axis=1) / var
        ST = 0.5 * np.mean((a[None, :] - ab) ** 2, axis=1) / var
        return S1, ST

    S1, ST = est(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = [est(rng.integers(0, n, n)) for _ in range(n_boot)]
    b1 = np.array([b[0] for b in boots])
    bt = np.array([b[1] for b in boots])
    return pd.DataFrame({"parameter": list(bounds), "S1": S1, "S1_low": np.quantile(b1, 0.05, axis=0),
                         "S1_high": np.quantile(b1, 0.95, axis=0), "ST": ST, "ST_low": np.quantile(bt, 0.05, axis=0),
                         "ST_high": np.quantile(bt, 0.95, axis=0)}).sort_values("ST", ascending=False)


@cites("morris1991")
def morris_screening(f: Callable[[np.ndarray], np.ndarray], bounds: dict[str, tuple[float, float]], r: int = 10,
                     levels: int = 4, seed: int = 0) -> pd.DataFrame:
    """Morris elementary effects: mu* (importance) and sigma (non-linearity/interaction)."""
    rng = np.random.default_rng(seed)
    d = len(bounds)
    lo = np.array([b[0] for b in bounds.values()])
    hi = np.array([b[1] for b in bounds.values()])
    delta = levels / (2 * (levels - 1))
    grid = np.arange(levels) / (levels - 1)
    EE = [[] for _ in range(d)]
    for _ in range(r):
        x = rng.choice(grid[grid <= 1 - delta + 1e-9], d)
        traj = [x.copy()]
        for i in rng.permutation(d):
            x = x.copy()
            x[i] = x[i] + delta if x[i] + delta <= 1 + 1e-9 else x[i] - delta
            traj.append(x)
        T = np.array(traj)
        y = f(lo + T * (hi - lo))
        for k in range(d):
            diff = T[k + 1] - T[k]
            i = int(np.flatnonzero(np.abs(diff) > 1e-12)[0])
            EE[i].append((y[k + 1] - y[k]) / diff[i])
    return pd.DataFrame({"parameter": list(bounds), "mu_star": [np.mean(np.abs(e)) for e in EE],
                         "mu": [np.mean(e) for e in EE], "sigma": [np.std(e) for e in EE]}).sort_values("mu_star", ascending=False)


def abm_response(cfg: SimConfig, params: list[str], metric: str = "profit", seeds: tuple[int, ...] = (0,),
                 workers: int = 1) -> Callable[[np.ndarray], np.ndarray]:
    """Turn the simulator into f(X) -> metric for Sobol/Morris (mean over seeds, common random numbers)."""
    def f(X: np.ndarray) -> np.ndarray:
        designs = [dict(zip(params, row)) for row in X]
        df = run_batch(cfg, designs, list(seeds), workers)
        return df.groupby(df.index // len(seeds))[metric].mean().to_numpy()
    return f


@dataclass
class ABResult:
    metric: str
    baseline_mean: float
    treatment_mean: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    n_reps: int

    def to_dict(self):
        return self.__dict__.copy()


def ab_simulation(cfg: SimConfig, treatment: dict, metrics=("ever_adopted", "profit", "roi", "active_customers"),
                  n_reps: int = 10, workers: int = 1, level: float = 0.9) -> tuple[list[ABResult], pd.DataFrame]:
    """Paired comparison with common random numbers (same seeds in both arms)."""
    seeds = list(range(cfg.seed, cfg.seed + n_reps))
    base = run_batch(cfg, [{}], seeds, workers).assign(arm="baseline")
    trt = run_batch(cfg, [treatment], seeds, workers).assign(arm="treatment")
    out = []
    for m in metrics:
        d = trt[m].to_numpy(dtype=float) - base[m].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        tq = stats.t.ppf(0.5 + level / 2, max(len(d) - 1, 1))
        p = float(stats.ttest_1samp(d, 0).pvalue) if len(d) > 1 and d.std() > 0 else float("nan")
        out.append(ABResult(m, float(base[m].mean()), float(trt[m].mean()), float(d.mean()),
                            float(d.mean() - tq * se), float(d.mean() + tq * se), p, len(d)))
    return out, pd.concat([base, trt], ignore_index=True)
