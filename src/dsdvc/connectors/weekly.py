"""Assemble connector outputs into the canonical ``marketing_weekly`` dataset.

Rules
* Weeks start on Monday; each source is bucketed in its own time zone (ad account, GA4 property,
  shop). Partial weeks at either end of the requested range are dropped.
* Outcome of record: Shopify net sales (orders as conversions) when available, else GA4
  ``purchaseRevenue`` (or ``totalRevenue``) and ``transactions``.
* Spend: Meta by channel rules; Google Ads cost from GA4 ``advertiserAdCost`` when linked;
  any additional spend tables (e.g. TV, OOH, influencer invoices) merged by week.
* Controls: Shopify discount rate → ``promo`` (discount depth). Web sessions are **not** used as
  controls (they are caused by marketing).
* Reconciliation: GA4 vs Shopify revenue per week; large or unstable gaps are flagged because
  they indicate tracking problems that also affect platform-reported ROAS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd


def to_weekly(daily: pd.DataFrame, value_cols: list[str], start: date, end: date, how: str = "sum") -> pd.DataFrame:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d.date >= pd.Timestamp(start)) & (d.date <= pd.Timestamp(end))]
    d["week"] = d.date - pd.to_timedelta(d.date.dt.weekday, unit="D")
    w = d.groupby("week")[value_cols].agg(how).reset_index().rename(columns={"week": "date"})
    return w


def complete_weeks(start: date, end: date) -> pd.DatetimeIndex:
    """Mondays of every full Monday-Sunday week inside [start, end]."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    first = s + pd.Timedelta(days=(7 - s.weekday()) % 7)
    last_sunday = e - pd.Timedelta(days=(e.weekday() + 1) % 7)
    last = last_sunday - pd.Timedelta(days=6)
    return pd.date_range(first, last, freq="W-MON") if last >= first else pd.DatetimeIndex([])


@dataclass
class BuildResult:
    data: pd.DataFrame
    column_sources: dict[str, str]
    reconciliation: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, dict] = field(default_factory=dict)

    def attestation_text(self) -> str:
        srcs = sorted({v.split(":")[0] for v in self.column_sources.values()})
        return ("Pulled via official APIs under the client's authorisation: " + ", ".join(srcs) +
                ". Aggregated weekly; contains no personal data.")


def build_weekly(start: date, end: date, *, meta: pd.DataFrame | None = None, ga4: pd.DataFrame | None = None,
                 ga4_ads_cost: pd.DataFrame | None = None, shopify: pd.DataFrame | None = None,
                 extra_spend: dict[str, pd.DataFrame] | None = None, outcome_source: str = "auto",
                 google_ads_channel: str = "google_ads") -> BuildResult:
    weeks = complete_weeks(start, end)
    out = pd.DataFrame({"date": weeks})
    src: dict[str, str] = {}
    prov: dict[str, dict] = {}
    warnings: list[str] = []
    if len(weeks) == 0:
        return BuildResult(out, src, None, ["The date range contains no complete Monday-Sunday week."])

    # outcome
    use = outcome_source
    if use == "auto":
        use = "shopify" if shopify is not None else ("ga4" if ga4 is not None else "none")
    if use == "shopify" and shopify is not None:
        w = to_weekly(shopify, ["net_sales", "orders", "discounts", "subtotal"], start, end)
        w["revenue"], w["conversions"] = w.net_sales, w.orders
        w["promo"] = (w.discounts / (w.subtotal + w.discounts).where(lambda s: s > 0)).fillna(0.0)
        out = out.merge(w[["date", "revenue", "conversions", "promo"]], on="date", how="left")
        src.update({"revenue": "shopify:net_sales", "conversions": "shopify:orders", "promo": "shopify:discount_rate"})
        prov["shopify"] = shopify.attrs.get("provenance", {})
        warnings += shopify.attrs.get("warnings", [])
    elif use == "ga4" and ga4 is not None:
        rev = "purchaseRevenue" if ga4.get("purchaseRevenue", pd.Series([0])).sum() > 0 else "totalRevenue"
        w = to_weekly(ga4, [rev, "transactions"], start, end)
        out = out.merge(w.rename(columns={rev: "revenue", "transactions": "conversions"}), on="date", how="left")
        src.update({"revenue": f"ga4:{rev}", "conversions": "ga4:transactions"})
        warnings.append("Outcome taken from GA4: web-tracked revenue usually under-counts true sales (consent, "
                        "ad blockers, offline orders). Prefer the commerce platform as the outcome of record.")
    else:
        warnings.append("No outcome source: connect Shopify or GA4, or upload sales.")
    if ga4 is not None:
        prov["ga4"] = ga4.attrs.get("provenance", {})

    # spend
    if meta is not None:
        piv = meta.pivot_table(index="date", columns="channel", values="spend", aggfunc="sum").reset_index()
        chans = [c for c in piv.columns if c != "date"]
        w = to_weekly(piv, chans, start, end)
        for c in chans:
            out = out.merge(w[["date", c]].rename(columns={c: f"spend_{c}"}), on="date", how="left")
            src[f"spend_{c}"] = "meta_marketing_api:spend"
        prov["meta"] = meta.attrs.get("provenance", {})
    if ga4_ads_cost is not None:
        w = to_weekly(ga4_ads_cost, ["advertiserAdCost"], start, end)
        out = out.merge(w.rename(columns={"advertiserAdCost": f"spend_{google_ads_channel}"}), on="date", how="left")
        src[f"spend_{google_ads_channel}"] = "ga4_data_api:advertiserAdCost"
    for name, df in (extra_spend or {}).items():
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        col = "spend" if "spend" in d else [c for c in d.columns if c != "date"][0]
        if d.date.dt.weekday.nunique() == 1 and (d.date.dt.weekday == 0).all():
            w = d[["date", col]]
        else:
            w = to_weekly(d, [col], start, end)
        out = out.merge(w.rename(columns={col: f"spend_{name}"}), on="date", how="left")
        src[f"spend_{name}"] = "user_upload:spend"
    for c in [c for c in out.columns if c.startswith("spend_")]:
        missing = int(out[c].isna().sum())
        # ad platforms return no rows when nothing was delivered, so a gap there is genuinely zero spend;
        # a gap in a manual upload may be missing data and needs confirming
        if missing and src.get(c, "").startswith("user_upload"):
            warnings.append(f"{c}: {missing} week(s) missing from the upload were set to 0 (confirm no spend).")
        out[c] = out[c].fillna(0.0)

    # reconciliation
    rec = None
    if shopify is not None and ga4 is not None:
        rev = "purchaseRevenue" if ga4.get("purchaseRevenue", pd.Series([0])).sum() > 0 else "totalRevenue"
        a = to_weekly(shopify, ["net_sales"], start, end)
        b = to_weekly(ga4, [rev], start, end)
        rec = a.merge(b, on="date", how="inner")
        rec = rec[rec.date.isin(weeks)]
        rec["ga4_to_shopify"] = rec[rev] / rec.net_sales.replace(0, np.nan)
        med = float(rec.ga4_to_shopify.median())
        unstable = float((abs(rec.ga4_to_shopify / med - 1) > 0.2).mean()) if med and med == med else float("nan")
        if med == med and (med < 0.7 or med > 1.1):
            warnings.append(f"GA4 records {med:.0%} of Shopify revenue (median week): tracking gap. Platform-reported "
                            "conversions and ROAS will be biased by a similar amount.")
        if unstable == unstable and unstable > 0.15:
            warnings.append(f"GA4/Shopify ratio moves by >20% in {unstable:.0%} of weeks: tracking changed over time "
                            "(tag changes, consent mode). Do not use GA4 as the outcome.")
    out = out[["date"] + [c for c in ("revenue", "conversions") if c in out] +
              sorted(c for c in out.columns if c.startswith("spend_")) + [c for c in ("promo",) if c in out]]
    return BuildResult(out, src, rec, warnings, prov)
