"""SYNTHETIC weekly marketing data with planted MMM parameters (for tests and the demo client).

``messy_client_export`` reproduces what real client files look like - different column names,
Indian number formatting (1,20,000), currency symbols and dd/mm/yyyy dates - to exercise the
onboarding mapper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsdvc.mmm.model import geometric_adstock, hill

TRUTH = {
    #               theta  K(×mean) S    beta (fraction of mean revenue at saturation)
    "meta_ads":      (0.30, 0.9, 1.6, 0.22),
    "google_search": (0.10, 1.1, 1.2, 0.18),
    "youtube":       (0.55, 1.2, 1.8, 0.10),
    "influencers":   (0.45, 0.8, 1.4, 0.12),
    "ooh":           (0.70, 1.3, 2.0, 0.05),
}
MEAN_WEEKLY_SPEND = {"meta_ads": 400_000, "google_search": 300_000, "youtube": 200_000, "influencers": 150_000, "ooh": 120_000}


def synthetic_mmm_data(weeks: int = 156, seed: int = 0, base_revenue: float = 6_000_000) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=weeks, freq="W-MON")
    df = pd.DataFrame({"date": dates})
    contrib = {}
    for c, (theta, K, S, beta) in TRUTH.items():
        m = MEAN_WEEKLY_SPEND[c]
        flights = (rng.random(weeks) < 0.75).astype(float)            # campaign on/off weeks
        level = m * np.exp(rng.normal(0, 0.45, weeks)) * flights
        if c == "ooh":
            level = np.repeat(rng.random(weeks // 6 + 1) < 0.5, 6)[:weeks] * m * 1.8
        df[c] = np.round(level, -3)
        x = df[c].to_numpy() / df[c][df[c] > 0].mean()
        contrib[c] = beta * hill(geometric_adstock(x, theta), K, S) * base_revenue
    t = np.arange(weeks)
    season = 0.08 * np.sin(2 * np.pi * t / 52.18) + 0.05 * np.cos(4 * np.pi * t / 52.18)
    df["price_index"] = 1 + 0.05 * np.sin(2 * np.pi * t / 26) + rng.normal(0, 0.02, weeks)
    df["promo"] = (rng.random(weeks) < 0.15).astype(int)
    base = base_revenue * (1 + 0.15 * t / weeks + season - 0.6 * (df["price_index"] - 1) + 0.06 * df["promo"])
    noise = rng.normal(0, 0.03 * base_revenue, weeks)
    df["revenue"] = np.round(base + sum(contrib.values()) + noise, 0)
    truth = {"params": TRUTH, "contribution": {c: float(v.sum()) for c, v in contrib.items()},
             "roi": {c: float(contrib[c].sum() / df[c].sum()) for c in TRUTH}}
    return df, truth


def _indian(n: float) -> str:
    s = f"{int(round(n)):d}"
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def messy_client_export(weeks: int = 156, seed: int = 0) -> pd.DataFrame:
    """The same data as a typical client spreadsheet export."""
    df, _ = synthetic_mmm_data(weeks, seed)
    out = pd.DataFrame({
        "Week Starting": df.date.dt.strftime("%d/%m/%Y"),
        "FB + IG Spend (₹)": ["₹" + _indian(v) for v in df.meta_ads],
        "Google Search Cost": df.google_search.map(_indian),
        "YT Budget INR": df.youtube.astype(int),
        "Influencer Payments": df.influencers.map(_indian),
        "Hoardings/OOH": df.ooh.astype(int),
        "Avg Selling Price Index": df.price_index.round(3),
        "Promo Week?": df.promo.map({1: "Yes", 0: "No"}),
        "Total Sales (INR)": df.revenue.map(_indian),
        "Notes": "",
    })
    return out
