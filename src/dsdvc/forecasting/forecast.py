"""Forecasting engine (spec §54): 7/14/30/60/90-day forecasts with uncertainty.

Per metric we fit a small model set and select by rolling-origin error (spec §34):
  * bass        (adoption-type series only) - Bass NLS on the history
  * holt_damped - damped-trend exponential smoothing with weekly multiplicative seasonality
  * naive_seasonal - last week's pattern
Intervals: residual bootstrap from rolling-origin errors at each horizon (empirical, not
Gaussian), so they widen honestly with horizon.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from dsdvc.core.literature import cites
from dsdvc.diffusion.bass import fit_bass_nls

HORIZONS = (7, 14, 30, 60, 90)
METRICS = ("adoptions", "engagement", "shares", "conversions", "churn", "revenue")


def _seasonal_index(y: np.ndarray, period: int = 7) -> np.ndarray:
    if len(y) < 2 * period:
        return np.ones(period)
    base = pd.Series(y).rolling(period, center=True, min_periods=1).mean().to_numpy()
    ratio = np.where(base > 0, y / np.maximum(base, 1e-9), 1.0)
    idx = np.array([np.nanmean(ratio[k::period]) for k in range(period)])
    idx = np.where(np.isfinite(idx) & (idx > 0), idx, 1.0)
    return idx / idx.mean()


def holt_damped(y: np.ndarray, h: int, period: int = 7) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    s = _seasonal_index(y, period)
    n = len(y)
    ys = y / s[np.arange(n) % period]

    def run(params, out=False):
        a, b, phi = params
        lvl, tr = ys[0], (ys[min(7, n - 1)] - ys[0]) / max(min(7, n - 1), 1)
        sse = 0.0
        for v in ys[1:]:
            f = lvl + phi * tr
            sse += (v - f) ** 2
            new = a * v + (1 - a) * f
            tr = b * (new - lvl) + (1 - b) * phi * tr
            lvl = new
        if out:
            return lvl, tr
        return sse

    r = optimize.minimize(run, [0.3, 0.05, 0.9], bounds=[(0.01, 0.99), (0.0, 0.5), (0.8, 0.995)], method="L-BFGS-B")
    lvl, tr = run(r.x, out=True)
    phi = r.x[2]
    steps = np.arange(1, h + 1)
    damp = np.cumsum(phi ** steps)
    fc = (lvl + damp * tr) * s[(n + steps - 1) % period]
    return np.maximum(fc, 0)


def naive_seasonal(y: np.ndarray, h: int, period: int = 7) -> np.ndarray:
    last = np.asarray(y[-period:], dtype=float)
    return np.tile(last, h // period + 1)[:h]


@cites("bass1969")
def bass_forecast(y: np.ndarray, h: int) -> np.ndarray:
    f = fit_bass_nls(np.asarray(y, dtype=float))
    t = np.arange(len(y) + 1, len(y) + h + 1)
    return f.per_period(t)


MODELS = {"holt_damped": holt_damped, "naive_seasonal": naive_seasonal, "bass": bass_forecast}


@cites("hyndman2021")
def rolling_origin_errors(y: np.ndarray, model, h: int, initial: int, step: int = 7) -> np.ndarray:
    """Matrix of forecast errors (origins x horizon)."""
    errs = []
    for o in range(initial, len(y) - h + 1, step):
        try:
            fc = model(y[:o], h)
        except Exception:  # noqa: BLE001
            continue
        errs.append(y[o:o + h] - fc)
    return np.array(errs)


@dataclass
class ForecastResult:
    metric: str
    model: str
    path: pd.DataFrame          # day-level forecast with lower/upper
    horizons: pd.DataFrame      # cumulative totals per horizon with intervals
    model_comparison: pd.DataFrame


class Forecaster:
    def __init__(self, horizons=HORIZONS, level: float = 0.9, seed: int = 0, min_history: int = 28):
        self.horizons = horizons
        self.level = level
        self.seed = seed
        self.min_history = min_history

    def forecast(self, y: np.ndarray, metric: str, adoption_like: bool | None = None) -> ForecastResult:
        y = np.asarray(y, dtype=float)
        H = max(self.horizons)
        adoption_like = metric in ("adoptions", "conversions") if adoption_like is None else adoption_like
        cands = ["holt_damped", "naive_seasonal"] + (["bass"] if adoption_like else [])
        eval_h = min(14, max(7, len(y) // 6))
        initial = max(self.min_history, len(y) // 2)
        comp = []
        errs_by = {}
        for name in cands:
            E = rolling_origin_errors(y, MODELS[name], eval_h, initial)
            if len(E) == 0:
                continue
            errs_by[name] = E
            comp.append({"model": name, "rolling_mae": float(np.mean(np.abs(E))),
                         "rolling_rmse": float(np.sqrt(np.mean(E ** 2))), "origins": len(E)})
        comp_df = pd.DataFrame(comp).sort_values("rolling_mae").reset_index(drop=True) if comp else pd.DataFrame()
        best = comp_df.iloc[0]["model"] if len(comp_df) else "naive_seasonal"
        fc = MODELS[best](y, H)
        # residual bootstrap: relative errors at each horizon step, extended by sqrt-horizon scaling
        rng = np.random.default_rng(self.seed)
        E = errs_by.get(best)
        if E is not None and len(E):
            # error at step k drawn from rolling-origin errors at step min(k, eval_h); beyond the
            # evaluated horizon, inflate by sqrt(k / eval_h) (random-walk growth assumption)
            k = np.minimum(np.arange(H), E.shape[1] - 1)
            rows = rng.integers(0, len(E), size=(2000, H))
            inflate = np.sqrt(np.maximum(1.0, (np.arange(H) + 1) / E.shape[1]))
            sims = np.maximum(fc[None, :] + E[rows, k[None, :]] * inflate[None, :], 0)
        else:
            sims = np.maximum(fc[None, :] * rng.lognormal(0, 0.3, (2000, 1)), 0)
        a = (1 - self.level) / 2
        # point forecast = median of the bias-corrected simulation (consistent with the interval)
        fc_bc = np.median(sims, axis=0)
        path = pd.DataFrame({"day_ahead": np.arange(1, H + 1), "forecast": fc_bc, "raw_model_forecast": fc,
                             "lower": np.quantile(sims, a, axis=0), "upper": np.quantile(sims, 1 - a, axis=0)})
        rows = []
        for h in self.horizons:
            tot = sims[:, :h].sum(axis=1)
            rows.append({"metric": metric, "horizon_days": h, "forecast_total": float(np.median(tot)),
                         "lower": float(np.quantile(tot, a)), "upper": float(np.quantile(tot, 1 - a)),
                         "model": best, "status": "PREDICTED"})
        return ForecastResult(metric, best, path, pd.DataFrame(rows), comp_df)

    def forecast_frame(self, df: pd.DataFrame, metrics=METRICS) -> pd.DataFrame:
        out = []
        for m in metrics:
            if m in df:
                out.append(self.forecast(df[m].to_numpy(), m).horizons)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
