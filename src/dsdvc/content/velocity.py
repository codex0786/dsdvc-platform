"""Early velocity features (spec §21).

    Velocity      = dEngagement / dt
    ShareVelocity = dShares / dt
    Acceleration  = dVelocity / dt
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsdvc.core.literature import cites

HOURS = (1, 3, 6, 12, 24)


def velocity_profile(times_h: np.ndarray, cumulative: np.ndarray) -> dict:
    t = np.asarray(times_h, dtype=float)
    c = np.asarray(cumulative, dtype=float)
    if len(t) < 2:
        return {"velocity": float(c[-1] / max(t[-1], 1e-9)) if len(t) else 0.0, "acceleration": 0.0, "trend": "insufficient_data"}
    v = np.diff(c) / np.diff(t)
    tv = (t[1:] + t[:-1]) / 2
    a = np.diff(v) / np.diff(tv) if len(v) > 1 else np.array([0.0])
    last_v, last_a = float(v[-1]), float(a[-1])
    rel = last_a * (tv[-1] - tv[0] if len(tv) > 1 else 1) / max(abs(last_v), 1e-9)
    trend = "accelerating" if rel > 0.15 else ("decelerating" if rel < -0.15 else "stable")
    return {"velocity": last_v, "acceleration": last_a, "mean_velocity": float(c[-1] / t[-1]), "trend": trend}


@cites("cheng2014")
def early_features(df: pd.DataFrame, horizon_h: int, prefix: str = "shares") -> pd.DataFrame:
    """Velocity features using only observations available at ``horizon_h`` hours."""
    hrs = [h for h in HOURS if h <= horizon_h and f"{prefix}_{h}h" in df.columns]
    if not hrs:
        raise ValueError(f"No early columns <= {horizon_h}h")
    out = pd.DataFrame(index=df.index)
    C = df[[f"{prefix}_{h}h" for h in hrs]].to_numpy(dtype=float)
    out[f"log_{prefix}_at_h"] = np.log1p(C[:, -1])
    out[f"{prefix}_velocity"] = C[:, -1] / hrs[-1]
    if len(hrs) >= 2:
        v = np.diff(C, axis=1) / np.diff(hrs)
        out[f"{prefix}_velocity_last"] = v[:, -1]
        out[f"log_{prefix}_velocity_last"] = np.log1p(np.clip(v[:, -1], 0, None))
        if v.shape[1] >= 2:
            tv = (np.array(hrs[1:]) + np.array(hrs[:-1])) / 2
            out[f"{prefix}_acceleration"] = (v[:, -1] - v[:, -2]) / (tv[-1] - tv[-2])
        out[f"{prefix}_growth_ratio"] = np.log1p(C[:, -1]) - np.log1p(C[:, 0])
    if "views" in df.columns:
        out[f"{prefix}_per_follower_at_h"] = C[:, -1] / np.maximum(df.get("followers", pd.Series(1, index=df.index)), 1)
    return out


def classify_trends(df: pd.DataFrame, horizon_h: int = 12, prefix: str = "shares") -> pd.Series:
    hrs = [h for h in HOURS if h <= horizon_h and f"{prefix}_{h}h" in df.columns]
    return df.apply(lambda r: velocity_profile(np.array(hrs), r[[f"{prefix}_{h}h" for h in hrs]].to_numpy())["trend"], axis=1)
