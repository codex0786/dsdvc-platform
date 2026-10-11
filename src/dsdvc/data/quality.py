"""DataQualityEngine (spec §36).

Checks missingness, duplicates, bot-like activity, outliers, API gaps, sampling bias,
survivorship bias, platform measurement differences and temporal mismatch, and combines
them into a DataQualityScore in [0, 1] that downstream predictions carry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WEIGHTS = {"missingness": 0.2, "duplicates": 0.1, "bot_like": 0.15, "outliers": 0.1, "api_gaps": 0.15,
           "sampling_bias": 0.1, "survivorship": 0.1, "measurement": 0.05, "temporal_mismatch": 0.05}


@dataclass
class QualityReport:
    checks: dict[str, dict] = field(default_factory=dict)
    score: float = 1.0

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{"check": k, **v} for k, v in self.checks.items()])


class DataQualityEngine:
    def __init__(self, time_col: str = "timestamp", id_col: str = "content_id"):
        self.time_col, self.id_col = time_col, id_col

    def evaluate(self, df: pd.DataFrame, population_reference: dict[str, float] | None = None,
                 group_col: str = "platform", expected_freq: str = "D") -> QualityReport:
        r = QualityReport()
        n = max(len(df), 1)
        # missingness on key metrics
        key = [c for c in ("views", "likes", "comments", "shares", "saves", "followers") if c in df]
        miss = float(df[key].isna().mean().mean()) if key else 0.0
        r.checks["missingness"] = {"value": miss, "penalty": min(1, miss * 2), "detail": {c: float(df[c].isna().mean()) for c in key}}
        dup = float(df.duplicated(subset=[self.id_col]).mean()) if self.id_col in df else float(df.duplicated().mean())
        r.checks["duplicates"] = {"value": dup, "penalty": min(1, dup * 5)}
        # bot-like: implausible engagement ratios (likes >> views, comments without likes, extreme follower/following)
        bot = np.zeros(len(df), dtype=bool)
        if {"likes", "views"} <= set(df.columns):
            bot |= (df["likes"] > df["views"] * 0.9).fillna(False).to_numpy()
        if {"comments", "likes"} <= set(df.columns):
            bot |= ((df["comments"] > 3 * df["likes"]) & (df["comments"] > 50)).fillna(False).to_numpy()
        if {"followers", "following"} <= set(df.columns):
            bot |= ((df["following"] > 5000) & (df["followers"] < 0.05 * df["following"])).fillna(False).to_numpy()
        r.checks["bot_like"] = {"value": float(bot.mean()) if len(bot) else 0.0, "penalty": min(1, float(bot.mean()) * 5) if len(bot) else 0.0,
                                "flagged_rows": np.flatnonzero(bot)[:50].tolist()}
        # outliers: robust z on log metrics
        out_frac = 0.0
        if "shares" in df:
            x = np.log1p(df["shares"].fillna(0).clip(lower=0))
            mad = np.median(np.abs(x - x.median())) * 1.4826 or 1.0
            z = (x - x.median()) / mad
            out_frac = float((np.abs(z) > 5).mean())
        r.checks["outliers"] = {"value": out_frac, "penalty": min(1, out_frac * 5)}
        # API gaps: missing periods in the time index
        gap_frac = 0.0
        if self.time_col in df and len(df) > 2:
            ts = pd.to_datetime(df[self.time_col]).dt.floor(expected_freq)
            full = pd.date_range(ts.min(), ts.max(), freq=expected_freq)
            gap_frac = float(1 - ts.nunique() / max(len(full), 1))
        r.checks["api_gaps"] = {"value": gap_frac, "penalty": min(1, max(0.0, gap_frac - 0.1) * 2),
                                "detail": "share of expected periods with no observations"}
        # sampling bias: total-variation distance between sample mix and a reference mix
        tv = 0.0
        if population_reference and group_col in df:
            samp = df[group_col].value_counts(normalize=True)
            keys = set(samp.index) | set(population_reference)
            tv = 0.5 * sum(abs(samp.get(k, 0) - population_reference.get(k, 0)) for k in keys)
        r.checks["sampling_bias"] = {"value": tv, "penalty": min(1, tv * 2), "detail": "TV distance vs reference mix" if population_reference else "no reference supplied"}
        # survivorship: share of items with zero engagement (if ~0, deleted/failed posts likely missing)
        surv = 0.0
        if "engagement" in df or {"likes", "shares"} <= set(df.columns):
            eng = df["engagement"] if "engagement" in df else df["likes"].fillna(0) + df["shares"].fillna(0)
            zero = float((eng == 0).mean())
            surv = 1.0 if (zero < 0.005 and n > 200) else 0.0
            r.checks["survivorship"] = {"value": zero, "penalty": 0.5 * surv,
                                        "detail": "almost no zero-engagement items: low performers may be missing (survivorship bias)" if surv else "ok"}
        else:
            r.checks["survivorship"] = {"value": None, "penalty": 0.0}
        # platform measurement differences: e.g. WhatsApp has no public views; mixing counts across platforms
        meas = 0.0
        if group_col in df and "views" in df:
            by = df.groupby(group_col)["views"].apply(lambda s: s.isna().mean())
            meas = float((by > 0.5).mean())
        r.checks["measurement"] = {"value": meas, "penalty": meas,
                                   "detail": "platforms define 'views'/'reach' differently; compare within platform"}
        # temporal mismatch: records in the future or timezone-naive mixes
        tm = 0.0
        if self.time_col in df:
            ts = pd.to_datetime(df[self.time_col], utc=True, errors="coerce")
            tm = float((ts > pd.Timestamp.now(tz="UTC")).mean() + ts.isna().mean())
        r.checks["temporal_mismatch"] = {"value": tm, "penalty": min(1, tm * 5)}
        r.score = float(np.clip(1 - sum(WEIGHTS[k] * v["penalty"] for k, v in r.checks.items()), 0, 1))
        return r
