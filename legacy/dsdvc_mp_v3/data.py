"""Empirical-data interface for DSDVC-MP.

The ABM should be calibrated from authorized/aggregate observations rather than private
consumer graphs. This module standardizes the observations we need and maps them to
model parameters.
"""
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "platform", "period", "exposures", "engagements", "shares",
    "adoptions", "churns"
]

@dataclass
class EmpiricalEstimate:
    platform: str
    share_rate: float
    engagement_rate: float
    adoption_rate: float
    churn_rate: float
    exposure_scale: float


def load_observations(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def estimate_platform_rates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for platform, g in df.groupby("platform"):
        exposure = max(g.exposures.sum(), 1)
        rows.append({
            "platform": platform,
            "share_rate": g.shares.sum() / exposure,
            "engagement_rate": g.engagements.sum() / exposure,
            "adoption_rate": g.adoptions.sum() / exposure,
            "churn_rate": g.churns.sum() / max(g.adoptions.sum(), 1),
            "exposure_scale": np.log1p(exposure),
        })
    return pd.DataFrame(rows)


def build_calibration_targets(df: pd.DataFrame) -> dict:
    rates = estimate_platform_rates(df)
    return {row.platform: row.drop("platform").to_dict() for _, row in rates.iterrows()}
