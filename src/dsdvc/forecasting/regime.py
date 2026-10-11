"""Regime detection (spec §55) via change points + diffusion-phase diagnostics.

Change points: PELT (Killick et al. 2012) with a Gaussian mean-shift cost on log1p series
and BIC-style penalty. Phase classification combines the Bass-implied cumulative share
F = N(t)/m_hat with growth/acceleration of a smoothed adoption series. Labels are
heuristic decision aids with the evidence shown, not ground truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsdvc.core.literature import cites
from dsdvc.diffusion.bass import fit_bass_nls

REGIMES = ("pre-diffusion", "early diffusion", "viral acceleration", "critical mass", "saturation",
           "decline", "fragmentation", "competitive disruption")


@cites("killick2012")
def pelt(y: np.ndarray, penalty: float | None = None, min_size: int = 7) -> list[int]:
    """Exact PELT for changes in mean (Gaussian cost, variance estimated globally)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2 * min_size:
        return []
    mad = np.median(np.abs(np.diff(y) - np.median(np.diff(y)))) * 1.4826
    sigma2 = max(mad ** 2 / 2, np.var(np.diff(y)) / 2, 1e-9)
    penalty = penalty if penalty is not None else 6 * np.log(n)  # conservative: fewer, larger shifts
    cs = np.concatenate([[0], np.cumsum(y)])
    cs2 = np.concatenate([[0], np.cumsum(y * y)])

    def cost(s, e):
        m = e - s
        tot = cs[e] - cs[s]
        return (cs2[e] - cs2[s] - tot * tot / m) / sigma2

    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    last = np.zeros(n + 1, dtype=int)
    R = [0]
    for t in range(min_size, n + 1):
        cands = [s for s in R if t - s >= min_size]
        if not cands:
            continue
        vals = [F[s] + cost(s, t) + penalty for s in cands]
        k = int(np.argmin(vals))
        F[t], last[t] = vals[k], cands[k]
        R = [s for s, v in zip(cands, vals) if v - penalty <= F[t]] + [s for s in R if t - s < min_size] + [t]
    cps, t = [], n
    while t > 0:
        s = last[t]
        if s > 0:
            cps.append(int(s))
        t = s
    return sorted(cps)


def detect_regime(daily: pd.DataFrame, adoption_col: str = "adoptions", active_col: str | None = None,
                  modularity: pd.Series | None = None, competitor_events: list[int] | None = None,
                  window: int = 14) -> dict:
    y = daily[adoption_col].to_numpy(dtype=float)
    n = len(y)
    cps = pelt(np.log1p(y))
    sm = pd.Series(y).rolling(7, min_periods=1).mean().to_numpy()
    growth = (sm[-1] - sm[-min(window, n)]) / max(sm[-min(window, n)], 1)
    prev_growth = (sm[-min(window, n)] - sm[-min(2 * window, n)]) / max(sm[-min(2 * window, n)], 1)
    accel = growth - prev_growth
    try:
        f = fit_bass_nls(y)
        F = float(y.sum() / f.m)
        peak_passed = n > f.peak_time + 7
    except Exception:  # noqa: BLE001
        F, peak_passed = float("nan"), False
    evidence = {"cumulative_share_F": F, "growth_14d": float(growth), "acceleration": float(accel),
                "change_points": cps, "bass_peak_passed": bool(peak_passed)}
    regime = "early diffusion"
    if competitor_events and cps and any(abs(c - e) <= 7 for c in cps[-2:] for e in competitor_events) and growth < -0.2:
        regime = "competitive disruption"
    elif modularity is not None and len(modularity) > 2 and modularity.iloc[-1] - modularity.iloc[0] > 0.05:
        regime = "fragmentation"
        evidence["modularity_change"] = float(modularity.iloc[-1] - modularity.iloc[0])
    elif active_col and active_col in daily and daily[active_col].iloc[-1] < 0.9 * daily[active_col].max() and growth < -0.1:
        regime = "decline"
    elif np.isfinite(F) and F < 0.025 and sm[-1] < 1.5 * max(sm[: max(1, n // 3)].mean(), 1):
        regime = "pre-diffusion"
    elif growth > 0.5 and accel > 0:
        regime = "viral acceleration"
    elif np.isfinite(F) and 0.10 <= F <= 0.5 and growth > -0.1 and not peak_passed:
        regime = "critical mass"
    elif (np.isfinite(F) and F > 0.6) or (peak_passed and growth < -0.1):
        regime = "saturation"
    return {"regime": regime, "evidence": evidence,
            "note": "Heuristic phase label from change points, Bass-implied cumulative share and growth; verify with domain judgement."}
