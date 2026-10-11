"""Early-warning system (spec §56) and platform concentration risk (spec §57)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Alert:
    kind: str
    severity: str   # info / warning / critical
    message: str
    evidence: dict

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _z_last(x: pd.Series, window: int = 28) -> float:
    hist = x.iloc[-window - 1:-1]
    sd = hist.std()
    return float((x.iloc[-1] - hist.mean()) / sd) if sd and sd > 0 else 0.0


def _trend(x: pd.Series) -> float:
    """OLS slope relative to mean level."""
    y = x.to_numpy(dtype=float)
    if len(y) < 3 or np.mean(np.abs(y)) == 0:
        return 0.0
    t = np.arange(len(y))
    return float(np.polyfit(t, y, 1)[0] * len(y) / np.mean(np.abs(y)))


def platform_dependency(conversions_by_platform: dict[str, float], threshold: float = 0.6) -> dict:
    tot = sum(conversions_by_platform.values())
    shares = {k: v / tot for k, v in conversions_by_platform.items()} if tot else {}
    top = max(shares.items(), key=lambda kv: kv[1]) if shares else (None, 0.0)
    hhi = float(sum(s * s for s in shares.values()))
    return {"shares": shares, "max_platform": top[0], "dependency": float(top[1]), "hhi": hhi,
            "threshold": threshold, "flag": bool(top[1] > threshold),
            "message": f"High platform concentration risk: {top[0]} drives {top[1]:.0%} of conversions" if top[1] > threshold else "ok"}


def scan(daily: pd.DataFrame | None = None, content: pd.DataFrame | None = None,
         creators_history: pd.DataFrame | None = None, modularity: pd.Series | None = None,
         conversions_by_platform: dict | None = None, competitor_events: list[int] | None = None,
         z_threshold: float = 3.0) -> list[dict]:
    alerts: list[Alert] = []
    if daily is not None and len(daily) > 30:
        if "shares" in daily:
            z = _z_last(daily["shares"])
            if z > z_threshold:
                alerts.append(Alert("viral_breakout", "info", f"Shares {z:.1f} SD above 28-day baseline", {"z": z}))
        if "churn" in daily:
            z = _z_last(daily["churn"])
            if z > z_threshold:
                alerts.append(Alert("churn_spike", "critical", f"Churn {z:.1f} SD above baseline", {"z": z}))
        if {"conversions", "engagement"} <= set(daily.columns):
            rate = (daily["conversions"] / daily["engagement"].clip(lower=1)).rolling(7, min_periods=3).mean()
            tr = _trend(rate.iloc[-28:])
            if tr < -0.3:
                alerts.append(Alert("conversion_deterioration", "warning",
                                    f"Conversion per engagement trending down ({tr:+.0%} over 28 days)", {"trend": tr}))
        if competitor_events and "adoptions" in daily:
            last_ev = max(competitor_events)
            if last_ev < len(daily) - 3:
                before = daily["adoptions"].iloc[max(0, last_ev - 14):last_ev].mean()
                after = daily["adoptions"].iloc[last_ev:].mean()
                if before > 0 and after / before < 0.75:
                    alerts.append(Alert("competitor_attack", "warning",
                                        f"Adoptions fell {1 - after / before:.0%} after competitor action at day {last_ev}",
                                        {"before": float(before), "after": float(after)}))
    if content is not None and len(content) >= 10 and "engagement" in content:
        c = content.sort_values("timestamp")
        er = (c["engagement"] / c["views"].clip(lower=1)) if "views" in c else c["engagement"]
        tr = _trend(er.iloc[-20:])
        if tr < -0.3:
            alerts.append(Alert("content_fatigue", "warning", f"Engagement rate on recent posts trending down ({tr:+.0%})",
                                {"trend": tr}))
    if creators_history is not None and len(creators_history):
        for cid, g in creators_history.sort_values("period").groupby("creator_id"):
            if len(g) >= 4:
                tr = _trend(g["engagement_rate"])
                if tr < -0.4:
                    alerts.append(Alert("influencer_decline", "warning", f"{cid}: engagement rate trend {tr:+.0%}",
                                        {"creator_id": cid, "trend": tr}))
    if modularity is not None and len(modularity) >= 3 and modularity.iloc[-1] - modularity.iloc[0] > 0.05:
        alerts.append(Alert("community_fragmentation", "warning",
                            f"Modularity rose by {modularity.iloc[-1] - modularity.iloc[0]:.2f}: audience is segregating",
                            {"start": float(modularity.iloc[0]), "end": float(modularity.iloc[-1])}))
    if conversions_by_platform:
        dep = platform_dependency(conversions_by_platform)
        if dep["flag"]:
            alerts.append(Alert("platform_dependency", "warning", dep["message"], dep))
    return [a.to_dict() for a in alerts]
