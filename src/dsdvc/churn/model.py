"""Churn model and segment breakdowns (spec §28).

    P(Churn) = sigma(a + b1 Dissatisfaction + b2 Competition + b3 PricePressure
                       - b4 Engagement - b5 CommunityFit)

Signs are *expected* by theory but estimated freely; the fitted table shows whether the
data agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsdvc.core.stats import fit_logit

FEATURES = ["dissatisfaction", "competition", "price_pressure", "engagement", "community_fit"]
EXPECTED_SIGN = {"dissatisfaction": +1, "competition": +1, "price_pressure": +1, "engagement": -1, "community_fit": -1}


class ChurnModel:
    def __init__(self, features: list[str] | None = None, l2: float = 1.0):
        self.features = features or FEATURES
        self.l2 = l2

    def fit(self, customers: pd.DataFrame, target: str = "churned") -> "ChurnModel":
        self.res = fit_logit(customers[self.features], customers[target].to_numpy(), l2=self.l2)
        return self

    def table(self) -> pd.DataFrame:
        t = self.res.table()
        t["expected_sign"] = [EXPECTED_SIGN.get(i, np.nan) for i in t.index]
        t["sign_agrees"] = np.sign(t["coef"]) == t["expected_sign"]
        return t

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.res.predict(df[self.features].to_numpy())


def churn_breakdown(customers: pd.DataFrame, by: str, churn_col: str = "churned") -> pd.DataFrame:
    """Observed churn rate by acquisition channel / platform / content / influencer / community,
    with Wilson 90% intervals."""
    from scipy import stats
    z = stats.norm.ppf(0.95)
    g = customers.groupby(by)[churn_col].agg(["sum", "count"]).rename(columns={"sum": "churned", "count": "customers"})
    p = g.churned / g.customers
    n = g.customers
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    g["churn_rate"], g["ci_low"], g["ci_high"] = p, centre - half, centre + half
    return g.sort_values("churn_rate", ascending=False)
