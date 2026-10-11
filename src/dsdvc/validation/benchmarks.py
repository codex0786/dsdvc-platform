"""Model benchmarks and ablation (spec §34, §60, §61).

The complex model must earn its complexity: every richer model is compared out-of-sample
(rolling origin / temporal holdout) against simpler baselines.

Adoption: Bass vs logistic growth vs ML (gradient boosting on lags) vs ABM (shape-calibrated).
Ablation: Bass -> +Networks -> +STEPPS -> +Recommendation -> +Cross-platform -> +Competition -> +Churn,
each variant re-calibrated on the training window and scored on the holdout.
"""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pandas as pd
from scipy import optimize

from dsdvc.core.literature import cites
from dsdvc.diffusion.abm import SimConfig, run_simulation
from dsdvc.diffusion.bass import fit_bass_nls, logistic_cumulative_fit
from dsdvc.validation.metrics import regression_report


def _bass(y, h):
    f = fit_bass_nls(y)
    return f.per_period(np.arange(len(y) + 1, len(y) + h + 1))


def _logistic(y, h):
    f = logistic_cumulative_fit(y)
    t = np.arange(len(y), len(y) + h + 1)
    cum = f["predict_cum"](t)
    return np.maximum(np.diff(cum), 0)


def _ml(y, h, lags: int = 7):
    from sklearn.ensemble import HistGradientBoostingRegressor
    y = np.asarray(y, float)
    if len(y) <= lags + 5:
        return np.full(h, y[-lags:].mean())
    X = np.array([np.r_[y[i - lags:i], i, y[:i].sum()] for i in range(lags, len(y))])
    m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, random_state=0).fit(X, y[lags:])
    hist = list(y)
    out = []
    for k in range(h):
        i = len(hist)
        x = np.r_[hist[-lags:], i, sum(hist)]
        v = max(float(m.predict(x[None, :])[0]), 0.0)
        out.append(v)
        hist.append(v)
    return np.array(out)


def make_abm_forecaster(cfg: SimConfig, n_seeds: int = 3) -> Callable:
    """ABM shape-calibrated forecaster: simulate a mean adoption curve, then fit market size M
    and time-scale s on the history: y(t) ~ M * shape(s t). Two free parameters, like logistic."""
    runs = []
    for k in range(n_seeds):
        c = copy.deepcopy(cfg)
        c.seed = cfg.seed + k
        runs.append(run_simulation(c).timeseries.new_adoptions.to_numpy(float))
    shape = np.mean(runs, axis=0)
    shape = shape / max(shape.sum(), 1)
    T = len(shape)

    def curve(t, s):
        return np.interp(t * s, np.arange(1, T + 1), shape, right=0.0)

    def fc(y, h):
        y = np.asarray(y, float)
        t = np.arange(1, len(y) + 1)

        def sse(s):
            base = curve(t, s)
            M = (base @ y) / max(base @ base, 1e-12)
            return float(((M * base - y) ** 2).sum())
        r = optimize.minimize_scalar(sse, bounds=(0.1, 10), method="bounded")
        base = curve(t, r.x)
        M = (base @ y) / max(base @ base, 1e-12)
        return np.maximum(M * curve(np.arange(len(y) + 1, len(y) + h + 1), r.x), 0)

    return fc


@cites("hyndman2021", "rand_rust2011")
def adoption_benchmark(y: np.ndarray, horizon: int = 14, initial: int | None = None, step: int = 7,
                       abm_cfg: SimConfig | None = None) -> pd.DataFrame:
    y = np.asarray(y, float)
    initial = initial or max(20, len(y) // 3)
    models = {"naive_last_week": lambda yy, h: np.full(h, yy[-7:].mean()), "bass": _bass,
              "logistic": _logistic, "ml_gbm": _ml}
    if abm_cfg is not None:
        models["dsdvc_abm"] = make_abm_forecaster(abm_cfg)
    rows = []
    for name, f in models.items():
        errs, ys, ps = [], [], []
        for o in range(initial, len(y) - horizon + 1, step):
            try:
                p = f(y[:o], horizon)
            except Exception:  # noqa: BLE001
                continue
            ys.append(y[o:o + horizon])
            ps.append(p)
        if ys:
            rep = regression_report(np.concatenate(ys), np.concatenate(ps))
            rows.append({"model": name, "origins": len(ys), **rep})
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


ABLATION_STEPS = ["bass_only", "+networks", "+stepps", "+recommendation", "+cross_platform", "+competition", "+churn"]


def _variant(cfg: SimConfig, level: int) -> SimConfig:
    """Mechanisms switched on cumulatively up to ``level`` (index into ABLATION_STEPS)."""
    c = copy.deepcopy(cfg)
    if level < 1:   # no social/network transmission: external (Bass p) + paid/creator posts only
        for p in c.platforms.values():
            p.social_reach, p.group_reach = 0.0, 0.0
        c.complex_share = 0.0
    if level < 2:   # content appeal neutral
        c.stepps = {k: 0.5 for k in c.stepps}
        c.stepps_form = "additive"
    if level < 3:
        for p in c.platforms.values():
            p.algo_reach = 0.0
    if level < 4:
        c.transmission = [[0.0] * 3 for _ in range(3)]
    if level < 5:   # single brand
        c.brands = [b for b in c.brands if b.is_focal]
    if level < 6:   # no churn
        c.churn_coefs = {**c.churn_coefs, "intercept": -30.0}
        c.recovery_rate = 0.0
    return c


@cites("rand_rust2011")
def ablation_report(observed_daily: np.ndarray, cfg: SimConfig, train_frac: float = 0.6,
                    grid=np.linspace(-6.0, -1.5, 10), seeds=(0, 1), metric_col: str = "new_adoptions") -> pd.DataFrame:
    """Each variant is calibrated (category_purchase_intercept and population scale) on the training
    window, then scored on the holdout. Incremental value = RMSE reduction vs the previous variant."""
    y = np.asarray(observed_daily, float)
    T = len(y)
    k = int(train_frac * T)
    rows = []
    for level, name in enumerate(ABLATION_STEPS):
        best = None
        for b in grid:
            sims = []
            for s in seeds:
                c = _variant(cfg, level)
                c.category_purchase_intercept = float(b)
                c.steps = T
                c.seed = s
                sims.append(run_simulation(c).timeseries[metric_col].to_numpy(float))
            m = np.mean(sims, axis=0)
            scale = (m[:k] @ y[:k]) / max(m[:k] @ m[:k], 1e-12)
            sse = float(((scale * m[:k] - y[:k]) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, b, scale, m)
        _, b, scale, m = best
        rep = regression_report(y[k:], scale * m[k:])
        rows.append({"variant": name, "calibrated_intercept": b, "scale": scale,
                     "train_rmse": float(np.sqrt(best[0] / k)), "holdout_rmse": rep["rmse"], "holdout_mae": rep["mae"]})
    df = pd.DataFrame(rows)
    df["incremental_rmse_reduction"] = -df["holdout_rmse"].diff()
    df["earns_complexity"] = df["incremental_rmse_reduction"].fillna(0) > 0
    return df
