"""Validation scorecard for canonical datasets (the gate between upload and modelling)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dsdvc.onboarding.templates import DatasetSpec

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    message: str
    fix: str = ""


@dataclass
class Scorecard:
    dataset: str
    rows: int
    checks: list[Check] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(c.status == FAIL for c in self.checks):
            return "BLOCKED"
        if any(c.status == WARN for c in self.checks):
            return "READY_WITH_WARNINGS"
        return "READY"

    @property
    def score(self) -> float:
        w = {PASS: 1.0, WARN: 0.6, FAIL: 0.0}
        return float(np.mean([w[c.status] for c in self.checks])) if self.checks else 0.0

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.checks])

    def add(self, name, status, message, fix=""):
        self.checks.append(Check(name, status, message, fix))


def spend_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("spend_")]


def validate(df: pd.DataFrame, spec: DatasetSpec) -> Scorecard:
    sc = Scorecard(spec.key, len(df))
    # required columns
    missing = [c.name for c in spec.columns if c.required and c.name not in df]
    sc.add("required columns", FAIL if missing else PASS,
           f"Missing: {', '.join(missing)}" if missing else "All required columns present",
           "Map the missing columns or add them to the file." if missing else "")
    for group in spec.one_of:
        have = [g for g in group if g in df]
        sc.add(f"one of {group}", PASS if have else FAIL, f"Using {have[0]}" if have else f"Need at least one of {group}",
               "" if have else "Provide revenue or conversions as the outcome.")
    if spec.channel_pattern:
        sp = spend_columns(df)
        sc.add("channel spend columns", PASS if sp else FAIL, f"{len(sp)} channel(s): {', '.join(c[6:] for c in sp)}" if sp else "No spend_ columns",
               "" if sp else "Map at least one spend column (e.g. 'FB Spend' → spend_meta_ads).")
    # dates
    if "date" in df:
        bad = int(df["date"].isna().sum())
        sc.add("dates readable", FAIL if bad > 0.05 * len(df) else (WARN if bad else PASS),
               f"{bad} unreadable date(s)" if bad else "All dates parsed", "Use YYYY-MM-DD or DD/MM/YYYY consistently." if bad else "")
        d = df["date"].dropna()
        dup = int(d.duplicated().sum()) if "geo" not in df else int(df.duplicated(["date", "geo"]).sum())
        sc.add("duplicate periods", FAIL if dup else PASS, f"{dup} duplicated period(s)" if dup else "No duplicates",
               "Aggregate to one row per week (per geo)." if dup else "")
        if len(d) > 2 and "geo" not in df:
            gaps = d.sort_values().diff().dt.days.dropna()
            mode = gaps.mode().iloc[0] if len(gaps) else 7
            irregular = int((gaps != mode).sum())
            freq = "weekly" if mode == 7 else ("daily" if mode == 1 else f"every {mode:.0f} days")
            sc.add("regular frequency", WARN if irregular or mode != 7 else PASS,
                   f"{freq}; {irregular} irregular gap(s)" if irregular else freq,
                   "MMM expects weekly data; aggregate daily data to weeks and fill missing weeks with 0 spend." if (irregular or mode != 7) else "")
    # history length
    n = len(df)
    sc.add("history length", FAIL if n < spec.min_rows else (WARN if n < spec.recommended_rows else PASS),
           f"{n} periods (minimum {spec.min_rows}, recommended {spec.recommended_rows})",
           "Add more history; with < 1 year seasonality cannot be separated from media effects." if n < spec.recommended_rows else "")
    # numeric ranges / missing
    for c in spec.columns:
        if c.name in df and c.dtype in ("number", "integer"):
            col = df[c.name]
            na = int(col.isna().sum())
            if na:
                sc.add(f"missing {c.name}", WARN if na <= 0.05 * n else FAIL, f"{na} missing value(s)",
                       "Fill or remove the affected weeks.")
            if c.min is not None and (col < c.min).any():
                sc.add(f"range {c.name}", FAIL, f"{int((col < c.min).sum())} value(s) below {c.min}", "Check for sign or unit errors.")
    for c in spend_columns(df) if spec.channel_pattern else []:
        col = df[c]
        if (col < 0).any():
            sc.add(f"negative {c}", FAIL, f"{int((col < 0).sum())} negative spend value(s)", "Spend must be ≥ 0 (refunds belong elsewhere).")
        na = int(col.isna().sum())
        if na:
            sc.add(f"missing {c}", WARN, f"{na} blank week(s) treated as zero spend", "Confirm blanks mean no spend.")
        nz = col.fillna(0)
        cv = nz.std() / nz.mean() if nz.mean() > 0 else 0
        if nz.sum() == 0:
            sc.add(f"variation {c}", FAIL, "Channel never has spend", "Remove the column.")
        elif cv < 0.15:
            sc.add(f"variation {c}", WARN, f"Spend barely varies (CV {cv:.2f}); effect will be prior-driven",
                   "Vary spend over time or run a lift test for this channel.")
    if spec.channel_pattern and len(spend_columns(df)) > 1:
        C = df[spend_columns(df)].fillna(0).corr()
        pairs = [(a, b, C.loc[a, b]) for i, a in enumerate(C.index) for b in C.columns[i + 1:] if abs(C.loc[a, b]) > 0.85]
        sc.add("channel collinearity", WARN if pairs else PASS,
               "; ".join(f"{a[6:]}~{b[6:]} r={r:.2f}" for a, b, r in pairs) if pairs else "Channels vary independently enough",
               "Effects of these channels cannot be separated well; plan a lift test or staggered flighting." if pairs else "")
    target = next((t for t in ("revenue", "conversions") if t in df), None)
    if target:
        y = df[target]
        z = int((y == 0).sum())
        if z:
            sc.add(f"zero {target}", WARN, f"{z} week(s) with zero {target}", "Check for missing data rather than true zeros.")
        med = y.median()
        mad = (y - med).abs().median() * 1.4826 or 1
        out = int(((y - med).abs() / mad > 5).sum())
        sc.add(f"outliers {target}", WARN if out else PASS, f"{out} extreme week(s)" if out else "No extreme weeks",
               "Add a promo/holiday flag for those weeks or confirm they are correct." if out else "")
    return sc
