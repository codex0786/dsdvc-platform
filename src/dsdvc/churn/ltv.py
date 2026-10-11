"""Customer lifetime value (spec §29).

    LTV_i = sum_{t=0}^{T} (Revenue_{i,t} - Cost_{i,t}) / (1 + r)^t

Plus retention-curve (cohort) LTV and LTV/CAC/ROI by segment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsdvc.core.literature import cites


@cites("gupta2006")
def discounted_ltv(revenue: np.ndarray, cost: np.ndarray | float = 0.0, r: float = 0.01) -> float:
    rev = np.asarray(revenue, dtype=float)
    c = np.broadcast_to(np.asarray(cost, dtype=float), rev.shape)
    t = np.arange(len(rev))
    return float(np.sum((rev - c) / (1 + r) ** t))


@cites("gupta2006")
def retention_ltv(margin_per_period: float, retention: float, r: float = 0.01, horizon: int = 36,
                  acquisition_cost: float = 0.0) -> float:
    """Expected LTV with constant retention probability per period (finite horizon)."""
    t = np.arange(horizon)
    return float(np.sum(margin_per_period * retention ** t / (1 + r) ** t) - acquisition_cost)


def ltv_by(customers: pd.DataFrame, by: str, revenue_col: str = "monthly_revenue", months_col: str = "months_active",
           margin: float = 0.6, r: float = 0.01, cac: dict | None = None) -> pd.DataFrame:
    """Per-customer realised discounted LTV aggregated by a segment column."""
    d = customers[customers[months_col] > 0].copy()
    d["ltv"] = [discounted_ltv(np.full(int(m), rev * margin), 0.0, r) for rev, m in zip(d[revenue_col], d[months_col])]
    g = d.groupby(by)["ltv"].agg(["mean", "median", "count", "sum"]).rename(
        columns={"mean": "ltv_mean", "median": "ltv_median", "count": "customers", "sum": "ltv_total"})
    se = d.groupby(by)["ltv"].std() / np.sqrt(g.customers)
    g["ltv_ci_low"], g["ltv_ci_high"] = g.ltv_mean - 1.645 * se, g.ltv_mean + 1.645 * se
    if cac:
        g["cac"] = [cac.get(k, np.nan) for k in g.index]
        g["ltv_to_cac"] = g.ltv_mean / g.cac
        g["roi"] = (g.ltv_mean - g.cac) / g.cac
    return g.sort_values("ltv_mean", ascending=False)
